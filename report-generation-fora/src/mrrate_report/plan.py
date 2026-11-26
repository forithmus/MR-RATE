"""Diagnostic plan of the MR writer (FORA plan rules with brain MRI wording).

The tiers are those of ``fora.reportgen.plan.diagnostic_plan.DiagnosticPlanBuilder``: per-class Platt curves fitted on
the TRAIN studies, confident = calibrated p >= 0.5 (never below the train Youden cutoff), possible = train Youden
cutoff up to the confident cutoff, at most 20 possible findings per study ranked by calibrated p. Each planned finding
is written as
    <name> (calibrated probability p, Nx the base rate of X%; <evidence description>) <evidence tokens>
and the plan starts with the list of MRI sequences of the study. The prompt text is in ``plan_templates.txt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from fora.reportgen.plan import diagnostic_plan as DP

TEMPLATES = Path(__file__).resolve().parent / "plan_templates.txt"
SEQUENCE_ORDER = ("t1", "t1ce", "t2", "flair", "swi", "t2star", "dwi", "adc", "pd", "stir", "tof", "other")


def sequences_text(kinds) -> str:
    from .evidence import SEQUENCE_NAMES
    present = sorted({SEQUENCE_ORDER[k] if isinstance(k, int) else str(k) for k in kinds}, key=SEQUENCE_ORDER.index)
    return ", ".join(SEQUENCE_NAMES[k] for k in present) if present else "not specified"


class MRPlanBuilder(DP.DiagnosticPlanBuilder):
    def __init__(self, calibration: str | Path, labels, tokenizer):
        super().__init__(calibration, labels, tokenizer, "all_classes", "ct", plan_possible_cap=20, plan_cap_rank="prob",
                         plan_prob=True, fimg_tokens=0, evidence_descriptions=True, templates=TEMPLATES)

    def plan_content(self, logits, location_descriptions: Mapping[int, str] | None = None,
                     finding_token_classes=None) -> str:
        """``location_descriptions``: class -> evidence description; key -1 holds the sequence list of the study.
        ``finding_token_classes``: mapping class -> number of evidence tokens (pooled + focus tokens)."""
        T = self.templates
        selection = self.plan_selection(logits)
        descriptions = dict(location_descriptions or {})
        sequences = descriptions.pop(-1, "not specified")
        counts = {int(k): int(v) for k, v in (finding_token_classes or {}).items()}
        calibrated = self.probabilities(logits).reshape(-1)

        def described(index: int) -> str:
            inside = f"calibrated probability {self.probability_text(index, float(calibrated[int(index)]))}"
            if descriptions.get(index):
                inside += f"; {descriptions[index]}"
            text = f"{self.names[index]} ({inside})"
            if index in counts:
                text += " " + DP.FINDING_PLACEHOLDER * counts[index]
            return text

        confident = "; ".join(described(i) for i in selection.confident) or "none"
        possible = "; ".join(described(i) for i in selection.possible) or "none"
        content = T["findings_instruction"] + T["sequences"].format(sequences=sequences) + T["plan_header"]
        content += T["confident"].format(confident=confident) + T["possible"].format(possible=possible)
        content += T["plan_footer"] + T["descriptions"]
        if counts:
            content += T["evidence_tokens"]
        return content
