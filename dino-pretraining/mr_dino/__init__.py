"""Volumetric DINOv3 training for aligned (coreg/atlas) MR-RATE studies."""

from .data import CropSpec, MRAlignedDINO3DDataset, collate_dino3d

__all__ = ["CropSpec", "MRAlignedDINO3DDataset", "collate_dino3d"]
