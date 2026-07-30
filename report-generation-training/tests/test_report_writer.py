from types import SimpleNamespace

import torch
from torch import nn

from mrrate_report_training.model import ReportWriter
from mrrate_report_training.targets import make_report_target


class TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text, **_):
        values = [2 + (ord(char) % 13) for char in text][:20] or [2]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))


class TinyLLM(nn.Module):
    def __init__(self, hidden=8, vocabulary=16):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.output = nn.Linear(hidden, vocabulary)
        self.lora_report = nn.Parameter(torch.tensor(0.1))
        self.active = "report"

    def set_adapter(self, value):
        self.active = value

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, logits_to_keep, **_):
        hidden = inputs_embeds[:, -logits_to_keep:] + self.lora_report
        return SimpleNamespace(logits=self.output(hidden))


def test_prefix_runs_once_and_report_adapter_receives_gradient():
    llm = TinyLLM()
    llm.embedding.weight.requires_grad_(False)
    llm.output.weight.requires_grad_(False)
    llm.output.bias.requires_grad_(False)
    model = ReportWriter(
        llm,
        TinyTokenizer(),
        torch.randn(3, 8),
        visual_dim=8,
        num_visual_queries=2,
        resampler_depth=1,
        resampler_heads=2,
        max_target_tokens=20,
    )
    calls = []
    handle = model.resampler.register_forward_hook(lambda *_: calls.append(1))
    output = model(
        torch.randn(5, 8),
        torch.tensor([[0.2, -0.4, 1.1]]),
        torch.tensor([0.5, 0.5, 0.5]),
        make_report_target(
            "x", ["There is no hemorrhage.", "A small infarct is present."]
        ),
    )
    output["loss"].backward()
    handle.remove()
    assert len(calls) == 1
    assert llm.lora_report.grad is not None
    assert torch.isfinite(llm.lora_report.grad)

