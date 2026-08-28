"""Create a tiny raw MR-RATE-atlas NIfTI tree for integration tests."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--studies", type=int, default=8)
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--shape", type=int, nargs=3, default=(8, 32, 32))
    args = parser.parse_args()

    d, h, w = args.shape
    z, y, x = np.indices((d, h, w), dtype=np.float32)
    affine = np.diag([0.5, 0.5, 1.0, 1.0]).astype(np.float32)
    for study in range(args.studies):
        image_dir = Path(args.out) / "batch00" / f"dummy_{study:03d}" / "atlas_img"
        image_dir.mkdir(parents=True, exist_ok=True)
        for sequence in range(args.sequences):
            center = np.array((d, h, w), dtype=np.float32) / 2
            center += np.array((0, sequence - 1, study % 3 - 1), dtype=np.float32)
            radius = (
                ((z - center[0]) / max(2, d / 5)) ** 2
                + ((y - center[1]) / max(3, h / 5)) ** 2
                + ((x - center[2]) / max(3, w / 5)) ** 2
            )
            volume = np.exp(-radius) * (0.45 + 0.2 * sequence)
            volume += np.sin((x + sequence * 3) / 9) * 0.04
            # MR-RATE's NIfTI loader converts (X,Y,Z) to model (D,H,W)=(Z,X,Y).
            nifti_array = np.ascontiguousarray(volume.transpose(1, 2, 0), dtype=np.float32)
            nib.save(
                nib.Nifti1Image(nifti_array, affine),
                image_dir / f"sequence_{sequence:02d}.nii.gz",
            )
    print(f"Wrote {args.studies} raw atlas studies under {args.out}")


if __name__ == "__main__":
    main()
