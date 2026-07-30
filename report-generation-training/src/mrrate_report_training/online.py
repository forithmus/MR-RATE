from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch


class OnlineSource:
    """Frozen upstream MR-RATE encoder source with exact full token bags."""

    def __init__(
        self, config: dict, device: torch.device, *, split: str = "train"
    ) -> None:
        upstream = Path(config["upstream_root"]).resolve()
        search_paths = [
            upstream,
            upstream / "scripts",
            upstream / "mr_rate",
            upstream / "vision_encoder",
        ]
        for path in reversed(search_paths):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        # Keep these paths for lazy imports inside build_encoder (notably
        # vision_encoder); this is a dedicated training process.
        from extract_features import _load_and_verify, build_encoder
        from mil_probe_online import build_dataset, encode_study
        from data_inference import collate_fn_infer
        from data import SPACE_TO_IMG_SUBDIR

        data = config["data"]
        encoder_config = config["encoder"]
        args = argparse.Namespace(
            weights_path=config["encoder_checkpoint"],
            encoder=encoder_config["name"],
            vjepa21_checkpoint=encoder_config.get("vjepa21_checkpoint"),
            chunk_size=int(encoder_config.get("chunk_size", 64)),
            fusion_mode=encoder_config["fusion_mode"],
            pooling_strategy=encoder_config["pooling_strategy"],
            extra_latent_projection=bool(
                encoder_config.get("extra_latent_projection", False)
            ),
            dim_latent=int(encoder_config["dim_latent"]),
            data_folder=data.get("data_folder"),
            jsonl_file=data["jsonl_file"],
            labels_file=data["labels_file"],
            splits_csv=data["splits_csv"],
            space=data.get("space", "native_space"),
            normalizer=data.get("normalizer", "zscore"),
            preprocessed_dir=data.get("preprocessed_dir"),
            use_preprocessed=bool(data.get("use_preprocessed", False)),
            cache_allow_mismatch=False,
        )
        if args.fusion_mode != "late":
            raise ValueError("Exact MR MIL/report tokens require fusion_mode=late")
        self.dataset = build_dataset(args, split)
        if len(self.dataset) == 0 and args.data_folder:
            zip_paths = sorted(Path(args.data_folder).glob("batch*/*.zip"))
            if zip_paths:
                self.dataset.samples = self._zip_samples(
                    zip_paths, self.dataset, SPACE_TO_IMG_SUBDIR
                )
                self.dataset._mrrate_zip_mode = True
                self.dataset._mrrate_original_getitem = self.dataset.__class__.__getitem__
                self.dataset.__class__ = self._zip_dataset_class(
                    self.dataset.__class__, SPACE_TO_IMG_SUBDIR
                )
                print(
                    f"[online] ZIP streaming enabled for {len(self.dataset)} "
                    f"{split} studies",
                    flush=True,
                )
        if len(self.dataset) == 0:
            raise ValueError(f"Online {split} source contains no eligible studies")
        self.subject_ids = [
            str(sample["subject_id"]) for sample in self.dataset.samples
        ]
        self.encoder, dim = build_encoder(args)
        if int(dim) != int(encoder_config["dim_latent"]):
            raise ValueError("Constructed encoder dimension differs from config")
        _load_and_verify(self.encoder, args.weights_path, strict_missing=True)
        try:
            visual = self.encoder.visual_transformer
            if hasattr(visual, "model") and hasattr(visual.model, "merge_and_unload"):
                visual.model.merge_and_unload()
        except Exception as error:
            print(f"[online] LoRA merge skipped: {error}", flush=True)
        self.encoder.to(device=device, dtype=torch.bfloat16)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self._encode_study = encode_study
        self._collate = collate_fn_infer
        self.device = device

    @staticmethod
    def _zip_samples(zip_paths, dataset, _space_mapping) -> list[dict]:
        samples = []
        for zip_path in zip_paths:
            subject_id = zip_path.stem
            if subject_id not in dataset.subject_to_sentences:
                continue
            sample = {
                "subject_id": subject_id,
                "zip_path": str(zip_path),
                "sentences": dataset.subject_to_sentences[subject_id],
            }
            if subject_id in dataset.subject_to_labels:
                sample["labels"] = dataset.subject_to_labels[subject_id]
            samples.append(sample)
        return samples

    @staticmethod
    def _zip_dataset_class(base_class, space_mapping):
        class ZipStreamingDataset(base_class):
            def __getitem__(self, index):
                sample = self.samples[index]
                image_subdir = space_mapping.get(self.space, "img")
                temp_parent = Path(
                    os.environ.get(
                        "MRRATE_ZIP_TEMP",
                        f"/tmp/mrrate_zip_{os.environ.get('SLURM_JOB_ID', 'local')}",
                    )
                )
                temp_parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
                    image_paths = []
                    with zipfile.ZipFile(sample["zip_path"]) as archive:
                        members = sorted(
                            value
                            for value in archive.namelist()
                            if value.endswith(".nii.gz")
                            and f"/{image_subdir}/" in f"/{value}"
                        )
                        if not members:
                            raise ValueError(
                                f"{sample['subject_id']} has no {image_subdir} NIfTIs"
                            )
                        for member_index, member in enumerate(members):
                            filename = f"{member_index:03d}_{Path(member).name}"
                            destination = Path(directory) / filename
                            with archive.open(member) as source, destination.open(
                                "wb"
                            ) as target:
                                shutil.copyfileobj(source, target)
                            image_paths.append(str(destination))
                    volumes = []
                    for image_path in image_paths:
                        volume = self.load_and_resample_nii(image_path)
                        volume = self.normalize_volume(volume)
                        volumes.append(self.crop_or_pad(volume))
                    stack = torch.stack(volumes, dim=0)
                mask = torch.ones(stack.shape[0], dtype=torch.bool)
                labels = sample.get("labels", np.array([], dtype=np.float32))
                return (
                    stack,
                    sample["sentences"],
                    sample["subject_id"],
                    mask,
                    labels,
                )

        ZipStreamingDataset.__name__ = "ZipStreamingMRReportDatasetInfer"
        return ZipStreamingDataset

    def __len__(self) -> int:
        return len(self.dataset)

    @torch.no_grad()
    def get(self, index: int) -> dict:
        batch = self._collate([self.dataset[index]])
        encoded = self._encode_study(
            self.encoder,
            batch,
            self.device,
            torch.bfloat16,
            0,
            keep_mapping=False,
        )
        if encoded.full_token_count != encoded.tokens.shape[0]:
            raise RuntimeError("Online token path unexpectedly capped a study")
        return {
            "subject_id": encoded.subject_id,
            "tokens": encoded.tokens,
            "mil_labels": encoded.target.squeeze(0).detach().cpu(),
            "series_count": encoded.series_count,
        }


def verify_frozen_encoder(source: OnlineSource) -> None:
    if source.encoder.training:
        raise RuntimeError("Online encoder must remain in eval mode")
    if any(parameter.requires_grad for parameter in source.encoder.parameters()):
        raise RuntimeError("Online encoder must remain frozen")
    if any(parameter.grad is not None for parameter in source.encoder.parameters()):
        raise RuntimeError("Frozen online encoder accumulated gradients")
