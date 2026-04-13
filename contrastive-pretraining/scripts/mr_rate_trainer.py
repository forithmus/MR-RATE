"""
MrRateTrainer: Distributed trainer for MR-RATE with brain MRI data.

Supports training on brain MRI with variable numbers of volumes per subject.
Each subject may have 2-12+ volumes (T1w, T2w, FLAIR, SWI, etc.).
Late fusion processes each volume independently, then pools across all.

Uses VL-CABS contrastive learning with sentence-level supervision.
"""

from pathlib import Path
from shutil import rmtree
from datetime import timedelta
import math
import re

from vision_encoder.optimizer import get_optimizer
from transformers import BertTokenizer

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from data import MRReportDataset, collate_fn, cycle

import numpy as np

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import torch.optim.lr_scheduler as lr_scheduler
from mr_rate import MRRATE

try:
    import wandb
except ImportError:
    wandb = None


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def yes_or_no(question):
    """Ask yes/no question. Returns False if not running interactively (e.g., batch job)."""
    import sys
    if not sys.stdin.isatty():
        print(f"{question} (y/n) -> Defaulting to 'n' (non-interactive mode)")
        return False
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup and cosine annealing restarts."""

    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))
        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma


class MrRateTrainer(nn.Module):
    """
    Distributed trainer for MR-RATE model.

    Data format per batch (batch_size=1):
    - images: [1, N, 1, D, H, W] - N volumes per subject (variable)
    - sentences: List of max_sentences strings
    - sentence_mask: [1, max_sentences] - which sentences are valid
    """

    def __init__(
            self,
            model: MRRATE,
            *,
            num_train_steps,
            batch_size=1,
            gradient_accumulation_steps=1,

            # MR dataset parameters
            data_folder=None,
            jsonl_file=None,
            splits_csv=None,
            split="train",
            space="native_space",
            normalizer="zscore",
            normalizer_kwargs=None,

            # Common dataset parameters
            max_sentences_per_image=34,

            # Training parameters
            tokenizer=None,
            lr=5e-5,
            wd=5e-2,
            max_grad_norm=0.5,
            warmup_steps=500,
            save_results_every=1000,
            save_model_every=1000,
            results_folder='./mr_rate_results/',
            num_workers=0,
            accelerate_kwargs: dict = dict(),

            # Resume
            resume=False,

            # W&B
            use_wandb=False,
            wandb_project="mr-rate",
            wandb_run_name=None,
            wandb_config=None,
    ):
        super().__init__()

        self.use_wandb = use_wandb and wandb is not None
        self.resume = resume

        # Initialize accelerator for distributed training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            **accelerate_kwargs
        )

        self.model = model

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                'microsoft/BiomedVLP-CXR-BERT-specialized',
                do_lower_case=True
            )

        self.register_buffer('steps', torch.Tensor([0]))
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.lr = lr

        # Initialize optimizer
        # IMPORTANT: use list, not set — set iteration order is non-deterministic
        # across runs, which causes optimizer state dict keys (positional indices)
        # to map to wrong parameters on resume, crashing with shape mismatches.
        all_parameters = list(model.parameters())
        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        # Initialize learning rate scheduler
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optim,
            T_0=num_train_steps,
            T_warmup=warmup_steps,
            eta_max=lr
        )

        # Initialize dataset
        self.print(f"[Trainer] Initializing MR dataset...")
        self.ds = MRReportDataset(
            data_folder=data_folder,
            jsonl_file=jsonl_file,
            max_sentences_per_image=max_sentences_per_image,
            space=space,
            normalizer=normalizer,
            normalizer_kwargs=normalizer_kwargs,
            splits_csv=splits_csv,
            split=split,
        )

        self.print(f"[Trainer] Dataset initialized with {len(self.ds)} subjects")
        self.print(f"[Trainer] Max sentences per image: {self.ds.max_sentences}")

        # Barrier: ensure all ranks have loaded the dataset before proceeding
        self.accelerator.wait_for_everyone()
        self.print(f"[Trainer] All ranks synchronized after dataset init")

        # Initialize dataloader
        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        # SyncBatchNorm conversion: convert on CPU first, then move to GPU
        self.device = self.accelerator.device
        if self.accelerator.num_processes > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.print("Converted BatchNorm to SyncBatchNorm")
        self.model.to(self.device)

        # Prepare for distributed training (device_placement=False since model is already on GPU)
        self.model = self.accelerator.prepare_model(self.model, device_placement=False)
        (
            self.dl,
            self.optim,
            self.scheduler
        ) = self.accelerator.prepare(
            self.dl,
            self.optim,
            self.scheduler
        )

        # Create infinite iterator AFTER prepare()
        self.dl_iter = cycle(self.dl)

        # Force optimizer LR to desired value
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        # Setup results folder
        self.results_folder = Path(results_folder)

        if self.is_main:
            if not self.resume and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no(
                'Do you want to clear previous experiment checkpoints and results?'
            ):
                rmtree(str(self.results_folder))
            self.results_folder.mkdir(parents=True, exist_ok=True)

        # Resume from latest checkpoint if requested
        if self.resume:
            self._auto_resume()

        # Initialize W&B on main process
        if self.use_wandb and self.is_main:
            resumed_step = int(self.steps.item())
            wandb_resume = "must" if (self.resume and resumed_step > 0) else "allow"
            run_id = None
            # Try to recover wandb run id from results folder for seamless resume
            wandb_id_file = self.results_folder / "wandb_run_id.txt"
            if self.resume and wandb_id_file.exists():
                run_id = wandb_id_file.read_text().strip()
                self.print(f"[W&B] Resuming run {run_id}")
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config or {},
                resume=wandb_resume,
                id=run_id,
            )
            # Save run id for future resumes
            wandb_id_file.write_text(wandb.run.id)

        # Synchronize all processes before training
        self.accelerator.wait_for_everyone()

    def _auto_resume(self):
        """Find and load the latest full checkpoint from results_folder."""
        # Look for full checkpoint files: MrRate.full.{step}.pt
        ckpts = sorted(
            self.results_folder.glob('MrRate.full.*.pt'),
            key=lambda p: int(re.search(r'\.full\.(\d+)\.pt$', p.name).group(1))
            if re.search(r'\.full\.(\d+)\.pt$', p.name) else 0
        )
        if not ckpts:
            self.print("[Resume] No full checkpoints found, starting from scratch")
            return
        latest = ckpts[-1]
        self.print(f"[Resume] Loading checkpoint: {latest}")
        self.load(latest)
        self.print(f"[Resume] Resumed at step {int(self.steps.item())}")

    def save(self, path):
        """Save full training checkpoint (model + optimizer + scheduler + step)."""
        if not self.accelerator.is_local_main_process:
            return
        pkg = dict(
            model=self.accelerator.get_state_dict(self.model),
            optim=self.optim.state_dict(),
            scheduler=self.scheduler.state_dict(),
            steps=self.steps.item()
        )
        torch.save(pkg, path)

    def load(self, path):
        """Load full training checkpoint."""
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"
        pkg = torch.load(path, map_location=self.device, weights_only=False)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])

        if 'scheduler' in pkg:
            self.scheduler.load_state_dict(pkg['scheduler'])
        if 'steps' in pkg:
            self.steps.fill_(pkg['steps'])

    def print(self, msg):
        """Print message on main process only."""
        self.accelerator.print(msg)

    @property
    def is_main(self):
        """Check if this is the main process."""
        return self.accelerator.is_main_process

    def train_step(self):
        """Execute a single training step."""
        device = self.device
        steps = int(self.steps.item())

        self.model.train()
        logs = {}

        if steps == 0:
            self.print(f"[DEBUG] dist.is_initialized={dist.is_initialized()}, accelerator.num_processes={self.accelerator.num_processes}")
            self.print(f"[DEBUG] Step {steps}: Loading batch...")

        # Get batch: [1, N, 1, D, H, W], sentences, [1, max_sentences]
        images, sentences, masks = next(self.dl_iter)

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Batch loaded. images={images.shape}, masks={masks.shape}")

        images = images.to(device)
        masks = masks.to(device)

        # Pad volumes to max across all ranks so DDP/SyncBatchNorm stay in sync
        n_volumes = images.shape[1]
        max_vols_t = torch.tensor([n_volumes], device=device)
        if self.accelerator.num_processes > 1:
            dist.all_reduce(max_vols_t, op=dist.ReduceOp.MAX)
        max_vols = int(max_vols_t.item())

        real_vols = torch.zeros(1, max_vols, device=device, dtype=torch.bool)
        real_vols[0, :n_volumes] = True

        if n_volumes < max_vols:
            pad_shape = list(images.shape)
            pad_shape[1] = max_vols - n_volumes
            images = torch.cat([images, torch.zeros(pad_shape, device=device, dtype=images.dtype)], dim=1)

        # Tokenize sentences
        tok = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Starting forward pass with {n_volumes} real volumes (padded to {max_vols})...")

        # Forward pass with mixed precision
        with self.accelerator.autocast():
            loss = self.model(
                text_input=tok,
                image=images,
                num_sentences_per_image=self.ds.max_sentences,
                sentence_mask=masks,
                real_volume_mask=real_vols,
                return_loss=True,
                device=device,
                debug=(steps == 0)
            )
            loss = loss / self.gradient_accumulation_steps

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Forward done, loss={loss.item():.4f}. Starting backward...")

        # NaN/Inf guard
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            self.print(f"[WARNING] Step {steps}: loss is {loss_val}, skipping backward/optimizer step")
            self.optim.zero_grad()
            accum_log(logs, {'loss': float('nan')})
            self.print(f"Step {steps}: loss=nan, lr={self.optim.param_groups[0]['lr']:.2e}")
            self.steps += 1
            return logs

        # Backward pass
        self.accelerator.backward(loss)

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Backward done.")

        # Accumulate loss for logging
        accum_log(logs, {'loss': loss.item() * self.gradient_accumulation_steps})

        # Update weights
        if (steps + 1) % self.gradient_accumulation_steps == 0:
            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            self.optim.step()
            self.optim.zero_grad()

        # Log progress
        current_lr = self.optim.param_groups[0]['lr']
        self.print(f"Step {steps}: loss={logs['loss']:.4f}, lr={current_lr:.2e}, n_vols={n_volumes}")

        # W&B logging
        if self.use_wandb and self.is_main:
            wandb.log({
                'train/loss': logs['loss'],
                'train/lr': current_lr,
                'train/n_volumes': n_volumes,
                'train/step': steps,
            }, step=steps)

        # Save checkpoints
        if self.is_main and steps > 0 and (steps % self.save_model_every == 0):
            # Model-only checkpoint (for inference)
            model_path = str(self.results_folder / f'MrRate.{steps}.pt')
            state_dict = self.accelerator.get_state_dict(self.model, unwrap=False)
            self.accelerator.save(state_dict, model_path)
            self.print(f"Saved model checkpoint: {model_path}")

            # Full checkpoint (for resume) — overwrite to save disk space
            full_path = str(self.results_folder / f'MrRate.full.{steps}.pt')
            self.save(full_path)
            self.print(f"Saved full checkpoint: {full_path}")

        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        """Run the full training loop."""
        start_step = int(self.steps.item())
        self.print(f"Starting training from step {start_step} to {self.num_train_steps}...")
        self.print(f"Batch size: {self.batch_size}, Grad accumulation: {self.gradient_accumulation_steps}")
        self.print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes}")

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        # Save final checkpoint
        if self.is_main:
            final_step = int(self.steps.item())
            final_model_path = str(self.results_folder / f'MrRate.{final_step}.pt')
            state_dict = self.accelerator.get_state_dict(self.model, unwrap=False)
            self.accelerator.save(state_dict, final_model_path)
            final_full_path = str(self.results_folder / f'MrRate.full.{final_step}.pt')
            self.save(final_full_path)
            self.print(f"Saved final checkpoints at step {final_step}")

        if self.use_wandb and self.is_main:
            wandb.finish()

        self.print('Training complete!')
