"""
dataloader.py — SARtoEO Dataset

Supports two dataset formats:
  1. SEN1-2  (mediatum.ub.tum.de/1436631)
     root/
       ROIs{id}_{season}/
         s1_{roi}/   <- SAR grayscale PNGs
         s2_{roi}/   <- EO RGB PNGs
     Split strategy: by season (spring+summer+fall=train, winter=val/test)

  2. Kaggle Sentinel-1&2 (terrain-segregated)
     root/
       {terrain}/
         s1/  <- SAR PNGs
         s2/  <- EO PNGs
     Split strategy: by terrain class (barren+grassland+agri=train, urban=val/test)

SAR Preprocessing:
  - Loaded as grayscale uint8 PNG [0, 255]
  - Normalised to [-1, 1] for network input
  - This matches the infer.py I/O contract (input is already [0, 255] dB-scaled)

EO Preprocessing:
  - Loaded as RGB uint8 PNG [0, 255]
  - Normalised to [-1, 1] for network input (tanh output)

Augmentation (train only, applied consistently to SAR+EO pair):
  - Random horizontal flip
  - Random vertical flip
  - Random 90° rotation
"""

import os
import glob
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Helper: consistent joint augmentation for SAR + EO pair
# ---------------------------------------------------------------------------

def joint_augment(sar: Image.Image, eo: Image.Image,
                  hflip: bool, vflip: bool, rot90: bool) -> Tuple[Image.Image, Image.Image]:
    """Apply the same random augmentation to both SAR and EO images."""
    if hflip and random.random() > 0.5:
        sar = TF.hflip(sar)
        eo  = TF.hflip(eo)
    if vflip and random.random() > 0.5:
        sar = TF.vflip(sar)
        eo  = TF.vflip(eo)
    if rot90:
        k = random.choice([0, 1, 2, 3])   # 0°, 90°, 180°, 270°
        if k > 0:
            sar = TF.rotate(sar, angle=90 * k)
            eo  = TF.rotate(eo,  angle=90 * k)
    return sar, eo


# ---------------------------------------------------------------------------
# Helper: image → normalised tensor
# ---------------------------------------------------------------------------

def to_tensor_normalised(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float tensor in [-1, 1]."""
    t = TF.to_tensor(img)   # [C, H, W], range [0, 1]
    t = t * 2.0 - 1.0       # → [-1, 1]
    return t


# ---------------------------------------------------------------------------
# SEN1-2 pair discovery
# ---------------------------------------------------------------------------

def _discover_sen12_pairs(root: str,
                           seasons: List[str]) -> List[Tuple[str, str]]:
    """
    Walk root directory, collect (sar_path, eo_path) pairs for the given seasons.

    SEN1-2 layout:
      root/ROIs{id}_{season}/s1_{roi}/ROIs{id}_{season}_s1_{roi}_p{patch}.png
      root/ROIs{id}_{season}/s2_{roi}/ROIs{id}_{season}_s2_{roi}_p{patch}.png
    """
    pairs: List[Tuple[str, str]] = []
    root_path = Path(root)

    for scene_dir in sorted(root_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        # Check if this scene belongs to one of the requested seasons
        scene_name = scene_dir.name.lower()           # e.g. "rois1158_spring"
        if not any(season in scene_name for season in seasons):
            continue

        # Find all s1_* subdirectories
        for s1_dir in sorted(scene_dir.glob("s1_*")):
            if not s1_dir.is_dir():
                continue
            # Corresponding s2 directory
            s2_dir = Path(str(s1_dir).replace("s1_", "s2_"))
            if not s2_dir.exists():
                continue

            for sar_path in sorted(s1_dir.glob("*.png")):
                # Derive matching EO path: replace 's1' with 's2' in filename
                eo_filename = sar_path.name.replace("_s1_", "_s2_")
                eo_path = s2_dir / eo_filename
                if eo_path.exists():
                    pairs.append((str(sar_path), str(eo_path)))

    return pairs


# ---------------------------------------------------------------------------
# Kaggle Sentinel-1&2 pair discovery
# ---------------------------------------------------------------------------

def _find_s1_s2_dir(terrain_dir: Path) -> Optional[Tuple[Path, Path]]:
    """Return (s1_dir, s2_dir) for a terrain directory, trying multiple naming conventions."""
    for s1_name, s2_name in [("s1", "s2"), ("SAR", "Optical"), ("sar", "optical"),
                              ("S1", "S2"), ("sen1", "sen2")]:
        s1 = terrain_dir / s1_name
        s2 = terrain_dir / s2_name
        if s1.exists() and s2.exists():
            return s1, s2
    return None


def _collect_pairs_from_dir(terrain_dir: Path) -> List[Tuple[str, str]]:
    """Collect (sar, eo) pairs from a single terrain directory."""
    result = _find_s1_s2_dir(terrain_dir)
    if result is None:
        return []
    s1_dir, s2_dir = result
    pairs = []
    # Support .tif, .tiff, .png, .jpg — satellite datasets commonly use .tif
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        for sar_path in sorted(s1_dir.glob(ext)):
            eo_path = s2_dir / sar_path.name
            if eo_path.exists():
                pairs.append((str(sar_path), str(eo_path)))
        if pairs:
            break  # Found matching pairs with this extension — stop searching
    return pairs



def _discover_kaggle_pairs(root: str,
                            terrains: List[str]) -> List[Tuple[str, str]]:
    """
    Walk root directory, collect (sar_path, eo_path) pairs for the given terrain classes.

    Supports multiple layouts:
      root/{terrain}/s1/*.png + root/{terrain}/s2/*.png  (standard)
      root/{terrain}/SAR/*.png + root/{terrain}/Optical/*.png  (alternate)

    If configured terrain names don't match any directory, falls back to
    scanning ALL subdirectories that contain valid s1/s2 pairs.
    """
    if root is None:
        raise ValueError(
            "kaggle_root is None — the Sentinel dataset is not mounted.\n"
            "In /kaggle/input, only these folders exist: "
            + str(list(Path("/kaggle/input").iterdir()) if Path("/kaggle/input").exists() else [])
            + "\nStop the session, add the Sentinel-1&2 dataset via '+ Add Input', then Run All."
        )
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"kaggle_root does not exist: {root}")

    pairs: List[Tuple[str, str]] = []

    # ---- Step 1: try configured terrain names (exact + case-insensitive) ----
    for terrain in terrains:
        terrain_dir = root_path / terrain
        if not terrain_dir.exists():
            matches = [d for d in root_path.iterdir()
                       if d.is_dir() and d.name.lower() == terrain.lower()]
            if not matches:
                print(f"[WARNING] Terrain dir '{terrain}' not found in {root}")
                continue
            terrain_dir = matches[0]
        pairs.extend(_collect_pairs_from_dir(terrain_dir))

    # ---- Step 2: fallback — scan ALL subdirs if no pairs found ---------------
    if not pairs:
        print(f"[INFO] Configured terrain names {terrains} not found. "
              f"Auto-scanning all subdirectories of {root}...")
        all_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        print(f"[INFO] Found subdirs: {[d.name for d in all_dirs]}")
        for terrain_dir in all_dirs:
            found = _collect_pairs_from_dir(terrain_dir)
            if found:
                print(f"[INFO] Found {len(found)} pairs in '{terrain_dir.name}'")
                pairs.extend(found)

    return pairs


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class SARtoEODataset(Dataset):
    """
    Paired SAR → EO dataset supporting SEN1-2 and Kaggle terrain formats.

    Args:
        cfg (dict):         Full config dict (from config.yaml)
        split (str):        'train', 'val', or 'test'
        augment (bool):     Apply augmentation (overrides split default)
    """

    def __init__(self, cfg: dict, split: str = "train",
                 augment: Optional[bool] = None):
        self.split = split
        self.augment = augment if augment is not None else (split == "train")

        data_cfg = cfg["data"]
        aug_cfg  = cfg.get("augmentation", {})

        self.hflip = aug_cfg.get("horizontal_flip", True)
        self.vflip = aug_cfg.get("vertical_flip", True)
        self.rot90 = aug_cfg.get("rotation_90", True)

        dataset_type = data_cfg.get("dataset_type", "sen12")
        subset_size  = data_cfg.get("subset_size", None)
        seed         = cfg.get("training", {}).get("seed", 42)

        # ---- collect pairs -----------------------------------------------
        if dataset_type == "sen12":
            root = data_cfg["sen12_root"]
            if split == "train":
                seasons = data_cfg.get("train_seasons", ["spring", "summer", "fall"])
            elif split == "val":
                seasons = data_cfg.get("val_seasons", ["winter"])
            else:  # test
                seasons = data_cfg.get("test_seasons", ["winter"])
            pairs = _discover_sen12_pairs(root, seasons)

        elif dataset_type == "kaggle":
            root = data_cfg["kaggle_root"]
            if split == "train":
                terrains = data_cfg.get("train_terrain", ["barren", "grassland", "agricultural"])
            elif split == "val":
                terrains = data_cfg.get("val_terrain", ["urban"])
            else:
                terrains = data_cfg.get("test_terrain", ["urban"])
            pairs = _discover_kaggle_pairs(root, terrains)

        else:
            raise ValueError(f"Unknown dataset_type: '{dataset_type}'. Use 'sen12' or 'kaggle'.")

        if not pairs:
            raise RuntimeError(
                f"No pairs found for split='{split}' in {root}. "
                f"Check your root_dir and season/terrain config."
            )

        # ---- subset sampling (deterministic, reproducible) ---------------
        if subset_size and subset_size < len(pairs):
            rng = random.Random(seed)
            pairs = rng.sample(pairs, subset_size)

        # For val/test on SEN1-2 (winter), split 50/50
        if dataset_type == "sen12" and split in ("val", "test"):
            rng = random.Random(seed)
            rng.shuffle(pairs)
            mid = len(pairs) // 2
            pairs = pairs[:mid] if split == "val" else pairs[mid:]

        self.pairs = pairs
        print(f"[Dataset] split={split} | {len(self.pairs):,} pairs | augment={self.augment}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sar_path, eo_path = self.pairs[idx]

        # Load images
        sar_img = Image.open(sar_path).convert("L")    # grayscale (1-channel)
        eo_img  = Image.open(eo_path).convert("RGB")   # RGB (3-channel)

        # Joint augmentation (same transform for both)
        if self.augment:
            sar_img, eo_img = joint_augment(
                sar_img, eo_img,
                hflip=self.hflip,
                vflip=self.vflip,
                rot90=self.rot90,
            )

        # Convert to normalised tensors in [-1, 1]
        sar_tensor = to_tensor_normalised(sar_img)   # [1, H, W]
        eo_tensor  = to_tensor_normalised(eo_img)    # [3, H, W]

        return {
            "sar":      sar_tensor,    # [1, 256, 256], float32, range [-1, 1]
            "eo":       eo_tensor,     # [3, 256, 256], float32, range [-1, 1]
            "sar_path": sar_path,
            "eo_path":  eo_path,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from config.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]

    batch_size  = train_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    train_ds = SARtoEODataset(cfg, split="train", augment=True)
    val_ds   = SARtoEODataset(cfg, split="val",   augment=False)
    test_ds  = SARtoEODataset(cfg, split="test",  augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml, sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    batch = next(iter(train_loader))
    print(f"SAR batch shape : {batch['sar'].shape}")   # [B, 1, 256, 256]
    print(f"EO  batch shape : {batch['eo'].shape}")    # [B, 3, 256, 256]
    print(f"SAR range       : [{batch['sar'].min():.2f}, {batch['sar'].max():.2f}]")
    print(f"EO  range       : [{batch['eo'].min():.2f}, {batch['eo'].max():.2f}]")
    print("Dataloader OK.")
