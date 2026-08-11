from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mrrate_report_training.generate import load_writer_checkpoint, shard_indices
from mrrate_report_training.model import ReportWriter, trainable_state_dict


VOCABULARY = 16


class TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text, **_):
        values = [2 + (ord(char) % 13) for char in text][:20] or [2]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, ids, **_):
        return " ".join(f"w{int(value)}" for value in ids)


class TinyLLM(nn.Module):
    """Cache-free decoder whose logits depend on the WHOLE sequence.

    The next-token distribution mixes the mean of every input embedding with
    the final embedding, so cached decoding that loses or duplicates context
    produces different text than full recomputation.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.embedding = nn.Embedding(VOCABULARY, 8)
        self.output = nn.Linear(8, VOCABULARY)
        self.lora_report = nn.Parameter(torch.tensor(0.1))
        for value in (*self.embedding.parameters(), *self.output.parameters()):
            value.requires_grad_(False)
        self.seen_masks = []

    def set_adapter(self, value):
        if value != "report":
            raise ValueError(value)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, logits_to_keep, **kwargs):
        mask = kwargs.get("attention_mask")
        if mask is not None:
            assert mask.shape[1] == inputs_embeds.shape[1]
        self.seen_masks.append(None if mask is None else mask.shape[1])
        context = inputs_embeds.sum(dim=1, keepdim=True) / inputs_embeds.shape[1]
        hidden = inputs_embeds[:, -logits_to_keep:] + context + self.lora_report
        return SimpleNamespace(logits=self.output(hidden))


class CachedTinyLLM(TinyLLM):
    """Same distribution as TinyLLM through a real running-state cache."""

    def forward(self, inputs_embeds, logits_to_keep, **kwargs):
        past = kwargs.get("past_key_values")
        mask = kwargs.get("attention_mask")
        if past is None:
            cached_length = inputs_embeds.shape[1]
            cached_sum = inputs_embeds.sum(dim=1, keepdim=True)
        else:
            assert inputs_embeds.shape[1] == 1
            cached_length = past["length"] + 1
            cached_sum = past["sum"] + inputs_embeds
        assert mask is not None and mask.shape[1] == cached_length
        self.seen_masks.append(mask.shape[1])
        context = cached_sum / cached_length
        hidden = inputs_embeds[:, -logits_to_keep:] + context + self.lora_report
        return SimpleNamespace(
            logits=self.output(hidden),
            past_key_values={"length": cached_length, "sum": cached_sum},
        )


def build_writer(llm) -> ReportWriter:
    torch.manual_seed(7)
    return ReportWriter(
        llm,
        TinyTokenizer(),
        torch.randn(3, 8),
        visual_dim=8,
        num_visual_queries=2,
        resampler_depth=1,
        resampler_heads=2,
        max_target_tokens=24,
    )


def study_inputs():
    torch.manual_seed(11)
    tokens = torch.randn(5, 8)
    mil_logits = torch.tensor([[0.2, -0.4, 1.1]])
    thresholds = torch.tensor([0.5, 0.5, 0.5])
    return tokens, mil_logits, thresholds


def test_generate_is_deterministic_and_bounded():
    writer = build_writer(TinyLLM()).eval()
    tokens, mil_logits, thresholds = study_inputs()
    first = writer.generate(tokens, mil_logits, thresholds, max_new_tokens=6)
    second = writer.generate(tokens, mil_logits, thresholds, max_new_tokens=6)
    assert first == second
    assert len(first.split()) <= 6


def test_generate_requires_eval_mode():
    writer = build_writer(TinyLLM()).train()
    tokens, mil_logits, thresholds = study_inputs()
    with pytest.raises(RuntimeError, match="eval mode"):
        writer.generate(tokens, mil_logits, thresholds)


def test_generate_stops_at_eos():
    class EOSLLM(TinyLLM):
        def forward(self, inputs_embeds, logits_to_keep, **kwargs):
            outputs = super().forward(inputs_embeds, logits_to_keep, **kwargs)
            outputs.logits = torch.zeros_like(outputs.logits)
            outputs.logits[..., TinyTokenizer.eos_token_id] = 10.0
            return outputs

    writer = build_writer(EOSLLM()).eval()
    tokens, mil_logits, thresholds = study_inputs()
    assert writer.generate(tokens, mil_logits, thresholds) == ""


def test_cached_and_uncached_decoding_match():
    plain_writer = build_writer(TinyLLM()).eval()
    cached_writer = build_writer(CachedTinyLLM()).eval()
    tokens, mil_logits, thresholds = study_inputs()
    plain = plain_writer.generate(tokens, mil_logits, thresholds, max_new_tokens=8)
    cached = cached_writer.generate(tokens, mil_logits, thresholds, max_new_tokens=8)
    assert plain == cached
    # The uncached model re-reads the growing full sequence; the cached model
    # sees single-token steps with a mask covering the full attended length.
    assert plain_writer.llm.seen_masks[-1] >= cached_writer.llm.seen_masks[-1]


def test_generate_prefix_matches_training_prefix():
    writer = build_writer(TinyLLM()).eval()
    tokens, mil_logits, thresholds = study_inputs()
    visual_prefix, mil_tokens = writer.shared_prefix(tokens, mil_logits, thresholds)
    prompt_ids = writer._token_ids(writer.REPORT_PROMPT, append_eos=False)
    expected = visual_prefix.shape[1] + mil_tokens.shape[1] + prompt_ids.numel()
    writer.llm.seen_masks.clear()
    writer.generate(tokens, mil_logits, thresholds, max_new_tokens=1)
    assert writer.llm.seen_masks[0] == expected


def test_load_writer_checkpoint_round_trip(tmp_path):
    writer = build_writer(TinyLLM())
    label_names = ["a", "b", "c"]
    package = {
        "trainable_state_dict": trainable_state_dict(writer),
        "label_names": label_names,
        "update": 3,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(package, path)
    fresh = build_writer(TinyLLM())
    with torch.no_grad():
        fresh.visual_projection[1].weight.add_(1.0)
    loaded = load_writer_checkpoint(path, fresh, label_names)
    assert int(loaded["update"]) == 3
    torch.testing.assert_close(
        fresh.visual_projection[1].weight, writer.visual_projection[1].weight
    )


def test_load_writer_checkpoint_rejects_schema_drift(tmp_path):
    writer = build_writer(TinyLLM())
    package = {
        "trainable_state_dict": trainable_state_dict(writer),
        "label_names": ["a", "b", "c"],
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(package, path)
    with pytest.raises(ValueError, match="label schema"):
        load_writer_checkpoint(path, build_writer(TinyLLM()), ["a", "b"])


def test_load_writer_checkpoint_rejects_missing_tensors(tmp_path):
    writer = build_writer(TinyLLM())
    state = trainable_state_dict(writer)
    state.pop(next(iter(state)))
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"trainable_state_dict": state, "label_names": ["a", "b", "c"]}, path
    )
    with pytest.raises(ValueError, match="trainable tensors differ"):
        load_writer_checkpoint(path, build_writer(TinyLLM()), ["a", "b", "c"])


def test_train_save_checkpoint_round_trips_into_inference_loader(tmp_path):
    """The real trainer checkpoint format must load through the inference path."""

    from mrrate_report_training.train import cosine_schedule, save_checkpoint

    writer = build_writer(TinyLLM())
    trainable = [value for value in writer.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    scheduler = cosine_schedule(optimizer, total_updates=2, warmup_ratio=0.0)
    label_names = ["a", "b", "c"]
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        writer,
        optimizer,
        scheduler,
        {"synthetic": True},
        label_names,
        epoch=0,
        next_slot=1,
        update=7,
        rank=0,
        world=1,
    )
    fresh = build_writer(TinyLLM())
    with torch.no_grad():
        fresh.image_start.add_(1.0)
    loaded = load_writer_checkpoint(path, fresh, label_names)
    assert int(loaded["update"]) == 7
    torch.testing.assert_close(fresh.image_start, writer.image_start)


def test_shard_indices_partition_everything_once():
    everything = sorted(
        index
        for shard in range(3)
        for index in shard_indices(10, 3, shard)
    )
    assert everything == list(range(10))
    with pytest.raises(ValueError):
        shard_indices(10, 3, 3)
