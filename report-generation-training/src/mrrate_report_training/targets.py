from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


_INLINE_SPACE = re.compile(r"[^\S\n]+")


def clean_findings(value: object) -> str:
    """Normalize whitespace while retaining report line boundaries."""

    lines = []
    for line in str(value or "").splitlines():
        cleaned = _INLINE_SPACE.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


@dataclass(frozen=True)
class ReportTarget:
    """Natural findings text from all_reports.csv; no inferred labels."""

    subject_id: str
    findings: str

    @property
    def text(self) -> str:
        return self.findings

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id cannot be empty")
        if not self.findings:
            raise ValueError(f"{self.subject_id}: findings cannot be empty")


def make_report_target(subject_id: str, findings: object) -> ReportTarget:
    target = ReportTarget(str(subject_id), clean_findings(findings))
    target.validate()
    return target


def load_target_index(path: str | Path) -> dict[str, ReportTarget]:
    """Load natural report findings keyed by study_uid."""

    targets: dict[str, ReportTarget] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"study_uid", "findings"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            subject_id = str(row["study_uid"]).strip()
            if not subject_id:
                raise ValueError(f"{path}:{line_number}: missing study_uid")
            if subject_id in targets:
                raise ValueError(f"{path}:{line_number}: duplicate {subject_id}")
            findings = clean_findings(row.get("findings"))
            if not findings:
                continue
            targets[subject_id] = ReportTarget(subject_id, findings)
    if not targets:
        raise ValueError(f"No report findings found in {path}")
    return targets
