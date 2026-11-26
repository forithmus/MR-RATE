"""74-diagnosis schema of the MR-RATE MIL and the readable names used in the writer plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPECIAL_NAMES = {
    "rathkes_cleft_cyst": "Rathke's cleft cyst",
    "cavernous_malformation_cavernoma": "cavernous malformation (cavernoma)",
    "moya_moya_disease": "moyamoya disease",
    "pachygyria_lissencephaly": "pachygyria / lissencephaly",
    "chordoma_chondrosarcoma": "chordoma / chondrosarcoma",
    "subdural_hygroma_effusions": "subdural hygroma / effusion",
    "encephalomalacia_gliosis": "encephalomalacia / gliosis",
    "craniotomy_craniectomy": "craniotomy / craniectomy",
    "dandy_walker_malformation": "Dandy-Walker malformation",
    "chiari_malformation": "Chiari malformation",
    "cavum_septum_pellucidum": "cavum septum pellucidum",
    "spine_syrinx": "spinal syrinx",
}


def readable(key: str) -> str:
    return SPECIAL_NAMES.get(key, key.replace("_", " "))


def label_order_sha256(labels) -> str:
    return hashlib.sha256("\n".join(map(str, labels)).encode()).hexdigest()


def checkpoint_classes(bundle: str | Path) -> list[str]:
    """Class order recorded with the MIL checkpoint (authoritative)."""
    return list(json.loads((Path(bundle) / "checkpoint/checkpoint_summary.json").read_text())["classes"])
