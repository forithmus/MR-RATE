"""Create a small MR-RATE-compatible coregistered cache for GPU smoke tests."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--studies", type=int, default=8)
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 192, 192))
    args = parser.parse_args()

    space = Path(args.out) / "coreg_space"
    space.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "layout": "per_subject_stack",
        "space": "coreg_space",
        "target_spacing": [1.0, 0.5, 0.5],
        "target_shape": list(args.shape),
        "posterior_shift_mm": 15.0,
        "normalizer": "zscore",
        "normalizer_kwargs": {},
        "dtype": "float16",
    }
    (space / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    z, y, x = np.indices(args.shape, dtype=np.float32)
    for study in range(args.studies):
        volumes = []
        for sequence in range(args.sequences):
            center = np.array(args.shape, dtype=np.float32) / 2
            center += np.array([0, sequence - 1, study % 3 - 1], dtype=np.float32)
            radius = (
                ((z - center[0]) / max(3, args.shape[0] / 5)) ** 2
                + ((y - center[1]) / max(3, args.shape[1] / 5)) ** 2
                + ((x - center[2]) / max(3, args.shape[2] / 5)) ** 2
            )
            volume = np.exp(-radius) * (0.45 + 0.2 * sequence)
            volume += np.sin((x + sequence * 3) / 9) * 0.04
            volumes.append(volume.astype(np.float16))
        np.savez(space / f"dummy_{study:03d}.npz", volumes=np.stack(volumes))
    print(f"Wrote {args.studies} dummy coregistered studies to {space}")


if __name__ == "__main__":
    main()
