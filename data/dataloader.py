"""
dataloader.py — SAR-to-EO Dataset with Combined SEN1-2 + Kaggle Support

Supports three dataset configurations:
  1. "sen12"    — SEN1-2 only (TU Munich, season-split)
  2. "kaggle"   — Kaggle Sentinel-1&2 only (terrain-split OR random)
  3. "combined" — Both datasets pooled together, random 80/10/10 split

Split strategies:
  "scene"   — group patches by source scene, then 80/10/10 (DEFAULT — leak-free)
  "terrain" — agri+barren+grassland=train, urban=val/test (shows generalisation)
  "random"  — 80/10/10 over individual patches (LEAKS on tiled data, see _scene_key)

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
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
# Scene grouping (leakage prevention)
# ---------------------------------------------------------------------------

# ROIs1970_fall_s1_13_p265.png → roi="ROIs1970_fall", scene="13", patch="265"
_SCENE_RE = re.compile(r"(ROIs\d+_[A-Za-z]+)_s[12]_(\d+)_p\d+", re.IGNORECASE)


def _scene_key(sar_path: str) -> str:
    """
    Return the source scene a patch was cut from, for use as a split group key.

    SEN1-2 (and the Kaggle set derived from it) tile each large scene on a fixed
    stride grid, so patches with neighbouring p-indices cover overlapping ground:
    p265, p266 and p267 are the same field shifted by one stride step. Splitting
    on individual patches therefore leaks — p265 trains the model and p266 tests
    it — which inflates PSNR/SSIM without any real generalisation.

    Grouping by scene keeps every tile of a scene on one side of the split.

    Falls back to the parent directory for filenames that do not follow the
    ROIs{id}_{season}_s{1,2}_{scene}_p{patch} convention.
    """
    m = _SCENE_RE.search(Path(sar_path).name)
    if m:
        return f"{m.group(1)}_scene{m.group(2)}"
    return str(Path(sar_path).parent)


# ---------------------------------------------------------------------------
# SEN1-2 pair discovery
# ---------------------------------------------------------------------------

def _discover_sen12_pairs(root: str,
                           seasons: List[str]) -> List[Tuple[str, str]]:
    """Walk SEN1-2 root directory, collect (sar_path, eo_path) pairs.

    Thin wrapper: forwards to a cached implementation keyed on (root,
    seasons) so `get_dataloaders()` building train/val/test datasets with
    identical args (e.g. combined mode's "all seasons" collection) does not
    re-walk the same directory tree 3x. Returns a fresh list each call so
    callers can freely mutate/reorder without corrupting the cache.
    """
    return list(_discover_sen12_pairs_cached(root, tuple(seasons)))


@lru_cache(maxsize=None)
def _discover_sen12_pairs_cached(root: str,
                                  seasons: Tuple[str, ...]) -> Tuple[Tuple[str, str], ...]:
    """
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
    return tuple(pairs)


# ---------------------------------------------------------------------------
# Kaggle Sentinel-1&2 pair discovery
# ---------------------------------------------------------------------------

def _discover_kaggle_pairs(root: str,
                            terrains: List[str]) -> List[Tuple[str, str]]:
    """Discover (sar_path, eo_path) pairs from the Kaggle Sentinel-1&2 dataset.

    Thin wrapper: forwards to a cached implementation keyed on (root,
    terrains) so `get_dataloaders()` building train/val/test datasets with
    identical args (e.g. combined/scene mode's "all terrains" collection)
    does not re-walk the same directory tree 3x. Returns a fresh list each
    call so callers can freely mutate/reorder without corrupting the cache.
    """
    if root is None:
        raise ValueError("kaggle_root is None — dataset not mounted.")
    return list(_discover_kaggle_pairs_cached(root, tuple(terrains or [])))


@lru_cache(maxsize=None)
def _discover_kaggle_pairs_cached(root: str,
                                   terrains: Tuple[str, ...]) -> Tuple[Tuple[str, str], ...]:
    """
    Supports many naming conventions for s1/s2 subdirectories.
    If terrains is empty, discovers ALL terrain subdirectories.
    """
    root_path = Path(root)
    if not root_path.exists():
        print(f"[WARNING] Kaggle root not found: {root} — skipping")
        return ()

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
            # D8: no filename overlap, only equal counts — if the two dirs
            # happen to sort differently (mixed extensions, a stray file,
            # different prefixes) this pairs SAR against the wrong EO image
            # for the WHOLE terrain, with no error. Print samples so a
            # mispair is at least visible, not purely silent.
            sample = ", ".join(f"{a.name}<->{b.name}"
                                for a, b in zip(s1_files[:3], s2_files[:3]))
            print(f"[WARNING] Pairing '{tdir.name}' by SORTED INDEX (no "
                  f"filename match) — verify these look right: {sample}")
            pairs.extend([(str(a), str(b)) for a, b in zip(s1_files, s2_files)])
        else:
            print(f"[WARNING] '{tdir.name}': count mismatch s1={len(s1_files)} "
                  f"vs s2={len(s2_files)} — skipping")

    print(f"[INFO] Kaggle total: {len(pairs)} pairs")
    return tuple(pairs)


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class SARtoEODataset(Dataset):
    """
    Paired SAR → EO dataset supporting SEN1-2, Kaggle, and combined loading.

    Dataset type and split strategy controlled via config.yaml:
      dataset_type: "sen12" | "kaggle" | "combined"
      split_strategy: "scene" | "terrain" | "random"

    Combined mode (dataset_type="combined"):
      Loads all pairs from both SEN1-2 and Kaggle datasets, pools them,
      then splits 80/10/10 by source scene. This maximises training data
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
        self.image_size = data_cfg.get("image_size", 256)

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
            # D1: only combined mode can have the SAME ground reachable under
            # two different keys (Kaggle set is derived from SEN1-2), so only
            # here must an unparseable scene id be fatal, not a warning.
            # That ambiguity needs BOTH roots to actually contribute, though —
            # gating on the mode alone bricks the Kaggle launch, where
            # dataset_type is "combined" but sen12_root is never mounted, so
            # the pool is single-root and provably unambiguous.
            strict = bool(sen12_pairs) and bool(kaggle_pairs)
            pairs = self._dispatch_split(all_pairs, split, split_strategy, seed,
                                         strict_scene_ids=strict)

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
        if split_strategy in ("scene", "random"):
            all_pairs = _discover_kaggle_pairs(root, [])  # all terrains
            return self._dispatch_split(all_pairs, split, split_strategy, seed)
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

    @classmethod
    def _dispatch_split(cls, pairs: List[Tuple[str, str]], split: str,
                        strategy: str, seed: int,
                        strict_scene_ids: bool = False) -> List[Tuple[str, str]]:
        """Route to the scene-disjoint or the legacy per-patch split."""
        if strategy == "scene":
            return cls._grouped_split(pairs, split, seed, strict_scene_ids)
        return cls._random_split(pairs, split, seed)

    @staticmethod
    def _grouped_split(pairs: List[Tuple[str, str]], split: str, seed: int,
                       strict_scene_ids: bool = False) -> List[Tuple[str, str]]:
        """
        Scene-disjoint 80/10/10 split.

        Assigns whole scenes rather than individual patches, so no scene spans
        two splits and no test patch overlaps ground the model saw during
        training. See _scene_key for why that matters.

        Each scene goes to whichever split is currently furthest below its
        target share. A plain "fill train to 80% then move on" pass would
        overshoot on the scene that crosses the boundary and can leave val or
        test empty, so val and test are seeded with one scene each first.

        Split sizes land near 80/10/10 rather than exactly on it, since scenes
        are indivisible and vary in patch count. The fewer scenes there are, the
        coarser the approximation.

        Args:
            strict_scene_ids: D1 — when the pool mixes two roots that can
                describe the SAME ground under different keys (combined mode:
                the Kaggle set is derived from SEN1-2), a directory-fallback
                key can no longer be trusted to keep splits disjoint — the
                fallback is safe only when every pair in the pool sources
                from filenames that use it consistently, which combined mode
                cannot guarantee. Set True to raise instead of warn.
        """
        groups: Dict[str, List[Tuple[str, str]]] = {}
        unparsed = 0
        first_unparsed = None
        for pair in pairs:
            key = _scene_key(pair[0])
            if not _SCENE_RE.search(Path(pair[0]).name):
                unparsed += 1
                if first_unparsed is None:
                    first_unparsed = pair[0]
            groups.setdefault(key, []).append(pair)

        # Falling back to the directory key means the filenames carry no scene
        # id. Splitting still cannot leak WITHIN one dataset root, but under
        # `combined` the same ground can be reachable through two roots with
        # two different keys (D1) — the fallback can no longer prove
        # disjointness, so that case must be fatal, not a warning.
        if unparsed:
            if strict_scene_ids:
                raise RuntimeError(
                    f"D1: {unparsed:,}/{len(pairs):,} paths have no parseable "
                    f"scene id (e.g. {first_unparsed!r} -> "
                    f"{_scene_key(first_unparsed)!r}) while dataset_type is "
                    f"'combined'. A directory-fallback key cannot prove this "
                    f"ground doesn't also appear under the OTHER root's "
                    f"ROIs-based key — see DATA_AUDIT.md §1c. Fix the "
                    f"filenames to the 'ROIs{{id}}_{{season}}_s{{1,2}}_"
                    f"{{scene}}_p{{patch}}' convention, or use dataset_type "
                    f"'sen12'/'kaggle' alone instead of 'combined'."
                )
            print(f"[Split] WARNING: {unparsed:,}/{len(pairs):,} paths have no "
                  f"parseable scene id; grouped by directory instead, e.g.:")
            print(f"           {Path(first_unparsed).name}  ->  "
                  f"{_scene_key(first_unparsed)}")
            print("         Splits stay leak-free but sizes will be uneven.")

        keys = sorted(groups)                 # sort first so seed alone decides
        random.Random(seed).shuffle(keys)

        if len(keys) < 3:
            raise RuntimeError(
                f"Only {len(keys)} scene group(s) found in {len(pairs)} pair(s) "
                f"— need at least 3 (one per split: train/val/test), and "
                f"practically ~6+ for a usable train share, since val and test "
                f"are each seeded with one scene first. Either the dataset is "
                f"too small, or filenames don't follow "
                f"'ROIs{{id}}_{{season}}_s{{1,2}}_{{scene}}_p{{patch}}' so "
                f"_scene_key fell back to grouping by directory (see the "
                f"[Split] WARNING above, if any). For a local sprint, use a "
                f"subset with >=6 distinct scenes (config: subset_size, "
                f"applied AFTER this split, so it cannot fix a too-small "
                f"source dataset — see DATA_AUDIT.md §5)."
            )

        n_total  = len(pairs)
        targets  = {"train": 0.80 * n_total, "val": 0.10 * n_total, "test": 0.10 * n_total}
        buckets: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}

        # Seed val and test with the two smallest scenes so neither starves and
        # the distortion of holding them back stays minimal.
        by_size = sorted(keys, key=lambda k: len(groups[k]))
        for name, key in (("val", by_size[0]), ("test", by_size[1])):
            buckets[name].extend(groups[key])

        seeded = {by_size[0], by_size[1]}
        for key in keys:
            if key in seeded:
                continue
            # Whichever split is furthest below its target share takes it.
            target = max(targets, key=lambda s: targets[s] - len(buckets[s]))
            buckets[target].extend(groups[key])

        print(f"[Split] scene-disjoint | {len(groups):,} scenes -> "
              f"train={len(buckets['train']):,} "
              f"val={len(buckets['val']):,} "
              f"test={len(buckets['test']):,}")

        for name, items in buckets.items():
            if not items:
                raise RuntimeError(
                    f"Split '{name}' came out empty from {len(groups)} scenes. "
                    f"Too few scenes to split three ways."
                )
        return buckets[split]

    @staticmethod
    def _random_split(pairs: List[Tuple[str, str]], split: str,
                      seed: int) -> List[Tuple[str, str]]:
        """
        80/10/10 split over individual patches.

        WARNING: leaks on tiled data. Adjacent tiles from one scene overlap on
        the ground, so shuffling patches puts near-duplicates on both sides of
        the split and inflates every metric. Kept only for reproducing older
        numbers — prefer "scene". See _scene_key.
        """
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

        # F3: data.image_size is otherwise enforced nowhere — any non-256px
        # pair reaches default_collate and crashes with an opaque stack-size
        # mismatch. Resize (no-op when already correct) instead.
        target = (self.image_size, self.image_size)
        if sar_img.size != target:
            sar_img = sar_img.resize(target, Image.BILINEAR)
        if eo_img.size != target:
            eo_img = eo_img.resize(target, Image.BICUBIC)

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

def get_dataloaders(
    cfg: dict,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Build train / val / test DataLoaders from config.

    Args:
        splits: which loaders to actually construct, e.g. ("train", "val")
                to skip building (and discovering/splitting for) an unused
                test loader — train.py only needs train+val, so building
                test every launch was pure waste (F8/D7).

    Returns:
        (train_loader, val_loader, test_loader) — entries for splits not
        requested are None. Always a 3-tuple so `a, b, _ = get_dataloaders(...)`
        keeps working regardless of `splits`.
    """
    train_cfg   = cfg["training"]
    data_cfg    = cfg["data"]
    batch_size  = train_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    # pin_memory only helps the CPU->CUDA copy; on a CPU-only box it's pure
    # overhead. persistent_workers avoids respawning worker processes (and
    # re-running dataset __init__, i.e. re-discovery) every single epoch.
    use_cuda   = torch.cuda.is_available()
    persistent = num_workers > 0

    loaders: Dict[str, Optional[DataLoader]] = {"train": None, "val": None, "test": None}

    if "train" in splits:
        train_ds = SARtoEODataset(cfg, split="train", augment=True)
        # D5: drop_last=True on a train split smaller than batch_size yields
        # ZERO batches every epoch — the loop runs, nothing trains, nothing
        # raises. Only drop the remainder when there's at least one full
        # batch to keep either way.
        train_drop_last = len(train_ds) >= batch_size
        if not train_drop_last:
            print(f"[DataLoader] WARNING: train split has only {len(train_ds)} "
                  f"pair(s) for batch_size={batch_size} — drop_last=True would "
                  f"silently yield ZERO batches every epoch. Disabling "
                  f"drop_last instead (final batch will be smaller).")
        loaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_cuda,
            drop_last=train_drop_last, persistent_workers=persistent,
        )

    if "val" in splits:
        val_ds = SARtoEODataset(cfg, split="val", augment=False)
        loaders["val"] = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=use_cuda,
            persistent_workers=persistent,
        )

    if "test" in splits:
        test_ds = SARtoEODataset(cfg, split="test", augment=False)
        loaders["test"] = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=use_cuda,
            persistent_workers=persistent,
        )

    return loaders["train"], loaders["val"], loaders["test"]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows consoles default to cp1252 and raise on the unicode used in the
    # progress output below. Force UTF-8 so local runs match Kaggle/Linux.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    import yaml, sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    batch = next(iter(train_loader))
    print(f"SAR shape : {batch['sar'].shape}")
    print(f"EO  shape : {batch['eo'].shape}")
    print(f"SAR range : [{batch['sar'].min():.2f}, {batch['sar'].max():.2f}]")
    print(f"EO  range : [{batch['eo'].min():.2f}, {batch['eo'].max():.2f}]")

    # ---- Leakage audit: no scene may appear in more than one split --------
    scenes = {
        name: {_scene_key(p[0]) for p in loader.dataset.pairs}
        for name, loader in [("train", train_loader),
                             ("val",   val_loader),
                             ("test",  test_loader)]
    }
    print("\n--- Leakage audit (scene overlap between splits) ---")
    leaked = False
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = scenes[a] & scenes[b]
        status = "OK" if not shared else f"LEAK - {len(shared)} shared scenes"
        print(f"  {a:<6} vs {b:<5} : {status}")
        leaked |= bool(shared)

    # D2: the check above compares _scene_key sets — the exact function that
    # silently fails under combined+renamed-Kaggle-copies (D1's fatal guard
    # covers that going forward, but this catches it independently, and also
    # catches same-ground double-counting when both copies DO keep matching
    # ROIs-based names, which is leak-free but not caught by D1). Compare raw
    # SAR filenames instead — orthogonal to _scene_key, so it can't share
    # _scene_key's blind spot.
    basenames = {
        name: [Path(p[0]).name for p in loader.dataset.pairs]
        for name, loader in [("train", train_loader),
                             ("val",   val_loader),
                             ("test",  test_loader)]
    }
    print("\n--- Basename overlap audit (independent of _scene_key) ---")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = set(basenames[a]) & set(basenames[b])
        status = "OK" if not shared else f"LEAK - {len(shared)} shared filenames"
        print(f"  {a:<6} vs {b:<5} : {status}")
        leaked |= bool(shared)
    for name, names in basenames.items():
        dup = len(names) - len(set(names))
        if dup:
            print(f"  {name:<6} : {dup} duplicate filename(s) within the split "
                  f"(same tile counted more than once)")

    print("\nDataloader OK - splits are scene-disjoint." if not leaked else
          "\nDataloader LEAKING - set split_strategy: \"scene\" in config.yaml.")
