"""
dataloader.py — SAR-to-EO Dataset with Combined SEN1-2 + Kaggle Support

Supports three dataset configurations:
  1. "sen12"    — SEN1-2 only (TU Munich, season-split)
  2. "kaggle"   — Kaggle Sentinel-1&2 only (terrain-split OR random)
  3. "combined" — Both datasets pooled together, random 80/10/10 split

Split strategies:
  "terrain" — agri+barren+grassland=train, urban=val/test (original, shows generalisation)
  "random"  — pool all data, 80/10/10 random split (best metrics, portfolio-optimal)

SAR Preprocessing:
  - Loaded as grayscale uint8 PNG [0, 255]
  - Normalised to [-1, 1] for network input

EO Preprocessing:
  - Loaded as RGB uint8 PNG [0, 255]
  - Normalised to [-1, 1] for network input (matches Tanh output)

Augmentation (train only, correctly applied ONCE per sample):
  - Random horizontal flip   (p=0.5)
  - Random vertical flip     (p=0.5)
  - Random 90° rotation      (uniform over 0/90/180/270)
  - SAR Gaussian noise       (σ ~ U[0, 0.05]) — simulates sensor noise variation
  - EO brightness jitter     (scale ~ U[0.9, 1.1]) — illumination variation
    Applied only to EO (not SAR), since EO brightness varies with acquisition time
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Helper: consistent joint augmentation (SAR + EO pair)
# ---------------------------------------------------------------------------

def joint_augment(sar: Image.Image, eo: Image.Image,
                  hflip: bool, vflip: bool, rot90: int,
                  sar_noise_std: float = 0.0,
                  eo_brightness: float = 1.0) -> Tuple[Image.Image, Image.Image]:
    """
    Apply the SAME geometric augmentation to both SAR and EO images.
    Apply SAR-specific noise and EO-specific brightness jitter separately.

    Random decisions are made BEFORE calling this function and passed as
    deterministic values — this guarantees exactly the intended probabilities.

    Args:
        sar, eo:          PIL Images
        hflip:            True → horizontal flip
        vflip:            True → vertical flip
        rot90:            0|1|2|3 — number of 90° CCW rotations
        sar_noise_std:    Std dev of Gaussian noise to add to SAR [0, 0.05]
        eo_brightness:    Brightness scale factor for EO [0.9, 1.1]
    """
    # Geometric (same for both)
    if hflip:
        sar = TF.hflip(sar)
        eo  = TF.hflip(eo)
    if vflip:
        sar = TF.vflip(sar)
        eo  = TF.vflip(eo)
    if rot90 > 0:
        sar = TF.rotate(sar, angle=90 * rot90)
        eo  = TF.rotate(eo,  angle=90 * rot90)

    # SAR-only: Gaussian noise (simulate sensor noise variation)
    if sar_noise_std > 0:
        sar_arr = np.array(sar, dtype=np.float32) / 255.0
        noise   = np.random.normal(0, sar_noise_std, sar_arr.shape).astype(np.float32)
        sar_arr = np.clip(sar_arr + noise, 0.0, 1.0)
        sar     = Image.fromarray((sar_arr * 255).astype(np.uint8))

    # EO-only: brightness jitter (illumination variation across acquisition times)
    if eo_brightness != 1.0:
        eo = TF.adjust_brightness(eo, eo_brightness)

    return sar, eo


# ---------------------------------------------------------------------------
# Helper: image → normalised tensor
# ---------------------------------------------------------------------------

def to_tensor_normalised(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float tensor in [-1, 1]."""
    t = TF.to_tensor(img)   # [C, H, W], [0, 1]
    t = t * 2.0 - 1.0       # → [-1, 1]
    return t


# ---------------------------------------------------------------------------
# SEN1-2 pair discovery
# ---------------------------------------------------------------------------

def _discover_sen12_pairs(root: str,
                           seasons: List[str]) -> List[Tuple[str, str]]:
    """
    Walk SEN1-2 root directory, collect (sar_path, eo_path) pairs.

    SEN1-2 layout:
      root/ROIs{id}_{season}/s1_{roi}/ROIs{id}_{season}_s1_{roi}_p{patch}.png
      root/ROIs{id}_{season}/s2_{roi}/ROIs{id}_{season}_s2_{roi}_p{patch}.png
    """
    pairs: List[Tuple[str, str]] = []
    root_path = Path(root)
    if not root_path.exists():
        print(f"[WARNING] SEN1-2 root not found: {root} — skipping")
        return pairs

    for scene_dir in sorted(root_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name.lower()
        if not any(season in scene_name for season in seasons):
            continue

        for s1_dir in sorted(scene_dir.glob("s1_*")):
            if not s1_dir.is_dir():
                continue
            s2_dir = Path(str(s1_dir).replace("s1_", "s2_"))
            if not s2_dir.exists():
                continue

            for sar_path in sorted(s1_dir.glob("*.png")):
                eo_filename = sar_path.name.replace("_s1_", "_s2_")
                eo_path = s2_dir / eo_filename
                if eo_path.exists():
                    pairs.append((str(sar_path), str(eo_path)))

    print(f"[INFO] SEN1-2 ({','.join(seasons)}): {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Kaggle Sentinel-1&2 pair discovery
# ---------------------------------------------------------------------------

def _discover_kaggle_pairs(root: str,
                            terrains: List[str]) -> List[Tuple[str, str]]:
    """
    Discover (sar_path, eo_path) pairs from the Kaggle Sentinel-1&2 dataset.

    Supports many naming conventions for s1/s2 subdirectories.
    If terrains is empty or None, discovers ALL terrain subdirectories.
    """
    if root is None:
        raise ValueError("kaggle_root is None — dataset not mounted.")
    root_path = Path(root)
    if not root_path.exists():
        print(f"[WARNING] Kaggle root not found: {root} — skipping")
        return []

    IMAGE_EXTS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    S1_NAMES   = ["s1", "sar", "sen1", "S1", "SAR", "sentinel1"]
    S2_NAMES   = ["s2", "optical", "sen2", "S2", "Optical", "sentinel2"]

    def find_s1_s2(tdir: Path):
        for s1n in S1_NAMES:
            for s2n in S2_NAMES:
                s1d, s2d = tdir / s1n, tdir / s2n
                if s1d.is_dir() and s2d.is_dir():
                    return s1d, s2d
        return None, None

    def glob_images(directory: Path) -> List[Path]:
        files = []
        for ext in IMAGE_EXTS:
            files.extend(directory.glob(ext))
        return sorted(files)

    all_subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    print(f"[INFO] Kaggle subdirs: {[d.name for d in all_subdirs]}")

    if terrains:
        terrain_lower = {t.lower() for t in terrains}
        terrain_dirs  = [d for d in all_subdirs if d.name.lower() in terrain_lower]
        if not terrain_dirs:
            print(f"[INFO] Requested terrains {terrains} not found — using all")
            terrain_dirs = all_subdirs
    else:
        terrain_dirs = all_subdirs

    pairs: List[Tuple[str, str]] = []
    for tdir in terrain_dirs:
        s1_dir, s2_dir = find_s1_s2(tdir)
        if s1_dir is None:
            for sub in sorted(tdir.iterdir()):
                if sub.is_dir():
                    s1_dir, s2_dir = find_s1_s2(sub)
                    if s1_dir is not None:
                        break
        if s1_dir is None:
            print(f"[WARNING] No s1/s2 dirs in '{tdir.name}' — skipping")
            continue

        s1_files = glob_images(s1_dir)
        s2_files = glob_images(s2_dir)
        print(f"[INFO]   {tdir.name}: s1={len(s1_files)}, s2={len(s2_files)}")

        if not s1_files or not s2_files:
            continue

        s2_map    = {f.name: f for f in s2_files}
        matched   = [(str(s1f), str(s2_map[s1f.name]))
                     for s1f in s1_files if s1f.name in s2_map]
        if matched:
            pairs.extend(matched)
        elif len(s1_files) == len(s2_files):
            print(f"[INFO] Pairing '{tdir.name}' by sorted index")
            pairs.extend([(str(a), str(b)) for a, b in zip(s1_files, s2_files)])
        else:
            print(f"[WARNING] '{tdir.name}': count mismatch s1={len(s1_files)} "
                  f"vs s2={len(s2_files)} — skipping")

    print(f"[INFO] Kaggle total: {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class SARtoEODataset(Dataset):
    """
    Paired SAR → EO dataset supporting SEN1-2, Kaggle, and combined loading.

    Dataset type and split strategy controlled via config.yaml:
      dataset_type: "sen12" | "kaggle" | "combined"
      split_strategy: "random" | "terrain"

    Combined mode (dataset_type="combined"):
      Loads all pairs from both SEN1-2 and Kaggle datasets, pools them,
      then does a random 80/10/10 split. This maximises training data
      diversity and is the recommended mode for best model quality.

    Args:
        cfg     (dict):  Full config dict (from config.yaml)
        split   (str):   'train', 'val', or 'test'
        augment (bool):  Override augmentation (default: True for train only)
    """

    def __init__(self, cfg: dict, split: str = "train",
                 augment: Optional[bool] = None):
        self.split   = split
        self.augment = augment if augment is not None else (split == "train")

        data_cfg  = cfg["data"]
        aug_cfg   = cfg.get("augmentation", {})

        self.hflip       = aug_cfg.get("horizontal_flip",    True)
        self.vflip       = aug_cfg.get("vertical_flip",      True)
        self.rot90       = aug_cfg.get("rotation_90",        True)
        self.sar_noise   = aug_cfg.get("sar_gaussian_noise", True)
        self.eo_bright   = aug_cfg.get("eo_brightness_jitter", True)

        dataset_type    = data_cfg.get("dataset_type",   "kaggle")
        split_strategy  = data_cfg.get("split_strategy", "random")
        subset_size     = data_cfg.get("subset_size",    None)
        seed            = cfg.get("training", {}).get("seed", 42)

        # ---- Collect pairs based on dataset_type -------------------------
        if dataset_type == "sen12":
            pairs = self._load_sen12(data_cfg, split)

        elif dataset_type == "kaggle":
            pairs = self._load_kaggle(data_cfg, split, split_strategy, seed)

        elif dataset_type == "combined":
            # Pool both datasets, then do a unified random split
            sen12_pairs  = self._collect_all_sen12(data_cfg)
            kaggle_pairs = self._collect_all_kaggle(data_cfg)
            all_pairs    = sen12_pairs + kaggle_pairs
            print(f"[Dataset] Combined: {len(sen12_pairs)} SEN1-2 + "
                  f"{len(kaggle_pairs)} Kaggle = {len(all_pairs)} total")
            pairs = self._random_split(all_pairs, split, seed)

        else:
            raise ValueError(f"Unknown dataset_type: '{dataset_type}'. "
                             f"Use 'sen12', 'kaggle', or 'combined'.")

        if not pairs:
            raise RuntimeError(
                f"No pairs found for split='{split}'. "
                f"Check your data paths in config.yaml."
            )

        # ---- Subset sampling (reproducible) ------------------------------
        if subset_size and subset_size < len(pairs):
            rng   = random.Random(seed)
            pairs = rng.sample(pairs, subset_size)

        self.pairs = pairs
        print(f"[Dataset] split={split} | {len(self.pairs):,} pairs | "
              f"augment={self.augment}")

    # ---- Private helpers --------------------------------------------------

    def _load_sen12(self, data_cfg: dict, split: str) -> List[Tuple[str, str]]:
        """Load SEN1-2 with season-based split."""
        root = data_cfg["sen12_root"]
        if split == "train":
            seasons = data_cfg.get("train_seasons", ["spring", "summer", "fall"])
        elif split == "val":
            seasons = data_cfg.get("val_seasons", ["winter"])
        else:
            seasons = data_cfg.get("test_seasons", ["winter"])
        pairs = _discover_sen12_pairs(root, seasons)
        # For SEN1-2, split winter 50/50 between val and test
        if split in ("val", "test"):
            pairs = self._split_half(pairs, split,
                                     cfg_seed=42)
        return pairs

    def _collect_all_sen12(self, data_cfg: dict) -> List[Tuple[str, str]]:
        """Load ALL SEN1-2 pairs (all seasons) for combined mode."""
        root    = data_cfg.get("sen12_root", "./data/SEN1-2")
        seasons = ["spring", "summer", "fall", "winter"]
        return _discover_sen12_pairs(root, seasons)

    def _load_kaggle(self, data_cfg: dict, split: str,
                     split_strategy: str, seed: int) -> List[Tuple[str, str]]:
        """Load Kaggle dataset with terrain or random split."""
        root = data_cfg["kaggle_root"]
        if split_strategy == "random":
            all_pairs = _discover_kaggle_pairs(root, [])  # all terrains
            return self._random_split(all_pairs, split, seed)
        else:
            # Terrain-segregated split
            if split == "train":
                terrains = data_cfg.get("train_terrain", ["agri", "barrenland", "grassland"])
            elif split == "val":
                terrains = data_cfg.get("val_terrain", ["urban"])
            else:
                terrains = data_cfg.get("test_terrain", ["urban"])
            return _discover_kaggle_pairs(root, terrains)

    def _collect_all_kaggle(self, data_cfg: dict) -> List[Tuple[str, str]]:
        """Load ALL Kaggle pairs (all terrains) for combined mode."""
        root = data_cfg.get("kaggle_root", "./data/sentinel12")
        return _discover_kaggle_pairs(root, [])

    @staticmethod
    def _random_split(pairs: List[Tuple[str, str]], split: str,
                      seed: int) -> List[Tuple[str, str]]:
        """80/10/10 random split."""
        rng = random.Random(seed)
        shuffled  = list(pairs)
        rng.shuffle(shuffled)
        n         = len(shuffled)
        train_end = int(n * 0.80)
        val_end   = int(n * 0.90)
        if split == "train":
            return shuffled[:train_end]
        elif split == "val":
            return shuffled[train_end:val_end]
        else:
            return shuffled[val_end:]

    @staticmethod
    def _split_half(pairs: List[Tuple[str, str]], split: str,
                    cfg_seed: int) -> List[Tuple[str, str]]:
        """50/50 split for SEN1-2 winter val/test."""
        rng = random.Random(cfg_seed)
        p   = list(pairs)
        rng.shuffle(p)
        mid = len(p) // 2
        return p[:mid] if split == "val" else p[mid:]

    # ---- Dataset interface ------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sar_path, eo_path = self.pairs[idx]

        sar_img = Image.open(sar_path).convert("L")    # 1-channel SAR
        eo_img  = Image.open(eo_path).convert("RGB")   # 3-channel EO

        if self.augment:
            # All random decisions made ONCE here — passed deterministically
            # to joint_augment to guarantee correct probabilities.
            do_hflip      = self.hflip and (random.random() < 0.5)
            do_vflip      = self.vflip and (random.random() < 0.5)
            do_rot90      = random.choice([0, 1, 2, 3]) if self.rot90 else 0
            noise_std     = random.uniform(0, 0.05) if self.sar_noise else 0.0
            eo_brightness = random.uniform(0.9, 1.1) if self.eo_bright else 1.0

            sar_img, eo_img = joint_augment(
                sar_img, eo_img,
                hflip=do_hflip, vflip=do_vflip, rot90=do_rot90,
                sar_noise_std=noise_std, eo_brightness=eo_brightness,
            )

        sar_tensor = to_tensor_normalised(sar_img)   # [1, H, W], [-1, 1]
        eo_tensor  = to_tensor_normalised(eo_img)    # [3, H, W], [-1, 1]

        return {
            "sar":      sar_tensor,
            "eo":       eo_tensor,
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
    train_cfg   = cfg["training"]
    data_cfg    = cfg["data"]
    batch_size  = train_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    train_ds = SARtoEODataset(cfg, split="train", augment=True)
    val_ds   = SARtoEODataset(cfg, split="val",   augment=False)
    test_ds  = SARtoEODataset(cfg, split="test",  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
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
    print(f"SAR shape : {batch['sar'].shape}")
    print(f"EO  shape : {batch['eo'].shape}")
    print(f"SAR range : [{batch['sar'].min():.2f}, {batch['sar'].max():.2f}]")
    print(f"EO  range : [{batch['eo'].min():.2f}, {batch['eo'].max():.2f}]")
    print("Dataloader OK. ✓")
