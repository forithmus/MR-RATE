import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
for p in (HERE / "src", HERE.parent / "dino-pretraining"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
