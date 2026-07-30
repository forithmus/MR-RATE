from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_SPACE = re.compile(r"\s+")


def clean_statement(value: object) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


@dataclass(frozen=True)
class ReportTarget:
    """All source findings in their original order, with no inferred labels."""

    subject_id: str
    statements: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.statements) if self.statements else "<NONE>"

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id cannot be empty")
        if any(not value for value in self.statements):
            raise ValueError(f"{self.subject_id}: report contains an empty statement")


def make_report_target(
    subject_id: str, statements: Iterable[object]
) -> ReportTarget:
    target = ReportTarget(
        str(subject_id),
        tuple(text for value in statements if (text := clean_statement(value))),
    )
    target.validate()
    return target


def load_target_index(path: str | Path) -> dict[str, ReportTarget]:
    targets: dict[str, ReportTarget] = {}
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            subject_id = clean_statement(
                row.get("volume_name") or row.get("study_uid") or row.get("subject_id")
            )
            if not subject_id:
                raise ValueError(f"{path}:{line_number}: missing study identifier")
            if subject_id in targets:
                raise ValueError(f"{path}:{line_number}: duplicate {subject_id}")
            statements = row.get("extracted_sentences")
            if not isinstance(statements, list):
                raise ValueError(
                    f"{path}:{line_number}: extracted_sentences must be a list"
                )
            targets[subject_id] = make_report_target(subject_id, statements)
    if not targets:
        raise ValueError(f"No report targets found in {path}")
    return targets

