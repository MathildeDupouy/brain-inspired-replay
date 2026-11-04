#!/usr/bin/env python3
"""
    Data Handler for HITS (Transcranial Doppler Ultrasound) dataset.

    Dataset: HITS Emboli — HITS-S51-13k-V9k
    Classes: A (Artefact), ES (Solid Emboli), EG (Gas Emboli)
    Format : HDF5 files with PNG paths stored as attributes

    Task A: data_2tasks-src-CLsplit_taskA.hdf5
            Train: 2817 samples | Test: 947 samples
    Task B: data_2tasks-src-CLsplit_taskB.hdf5
            Train: 868 samples  | Test: 139 samples

    Split strategy:
        - Train/Test splits are pre-defined in the HDF5 files.
        - Validation set is carved out from the training split
          using a stratified 85/15 split (same as supervisor protocol).
"""

import os
import random
import sys
from collections import Counter

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset

# For internal imports
sys.path.append(os.path.abspath(os.path.join("..")))



# =============================================================================
# Global handler for HITS dataset
# =============================================================================

class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_class_weights(self, dataset):
        if len(dataset[0]) == 2:
            labels = [y for _, y in dataset]
        else:
            labels = [y for _, y, _ in dataset]
        labels = np.array(labels).flatten()
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

# =============================================================================
# HITS Dataset class
# =============================================================================

class HITSDataset(Dataset):
    """
    PyTorch Dataset for HITS TCD Doppler spectrograms.

    Reads sample metadata (class label + PNG path) from an HDF5 file
    and loads images lazily from disk on __getitem__.

    Args:
        hdf5_path : path to the HDF5 file with global train_test splits
        split     : "train" or "test" (HDF5 group key)
        transform : torchvision transform pipeline
    """

    #----- Class attributes and methods -----
    CLASSES     = ["A", "ES", "EG"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    # Canonical HITS task definitions (used as stable reference for reordering)
    BASE_HITS_TASK_SRC = [(6,), (5,), (9,), (1, 3, 11,), (2,), (4, 8,)] #, (7, 10,)]
    HITS_TASK_SRC = list(BASE_HITS_TASK_SRC)
    # Stable colors per task identity (keyed by source-group, independent of current order)
    TASK_COLOR_BY_SRC = {
        (6,): "tab:blue",
        (5,): "tab:orange",
        (9,): "tab:green",
        (1, 3, 11): "tab:red",
        (2,): "tab:purple",
        (4, 8): "tab:brown",
    }
    # Explication
    # 6:                    15 sub, 4935 total, mainly A/GE, some FOP
    # 5:                    11 sub, 1969 total, all 3 classes
    # 9:                     1 sub,  609 total, mainly A/SE, stenosis only
    # TAVI group 1-3-11:     5 sub,  396 total, mainly A/GE, TAVI
        # 1:  3 sub, 291 total, GE only
        # 3:  1 sub,  94 total, mainly GE
        # 11: 1 sub,  11 total, A only
    # 2:                     8 sub,  613 total, mainly A/SE
    # stenosis group 4-8:    7 sub,  413 total, mainly A/SE, stenosis
        # 4:  1 sub,  305 total, mainly A/SE, stenosis
        # 8:  6 sub,  108 total, mainly SE/A, stenosis only
    # remaining group 7-10:  2 sub,  139 total, mainly A/SE, stenosis only
        # 7:  1 sub,   15 total, GE only
        # 10: 1 sub,   40 total, mainly A/GE, AFA
    @classmethod
    def set_task_order(cls, order):
        """Set a new task order for the HITS benchmark."""
        if len(order) != len(cls.BASE_HITS_TASK_SRC):
            raise ValueError(f"Task order must have length {len(cls.BASE_HITS_TASK_SRC)}")
        if set(order) != set(range(len(cls.BASE_HITS_TASK_SRC))):
            raise ValueError(f"Task order must contain all task indices 0-{len(cls.BASE_HITS_TASK_SRC)-1}")
        cls.HITS_TASK_SRC = [cls.BASE_HITS_TASK_SRC[i] for i in order]
        print(f"New HITS task order set: {cls.HITS_TASK_SRC}")

    @classmethod
    def get_task_colors(cls):
        """Return plotting colors aligned with the current HITS task order."""
        colors = []
        for task_src in cls.HITS_TASK_SRC:
            key = tuple(sorted(task_src))
            if key not in cls.TASK_COLOR_BY_SRC:
                raise KeyError(f"No color configured for HITS task source-group: {task_src}")
            colors.append(cls.TASK_COLOR_BY_SRC[key])
        return colors

    #----- Instance attributes and methods -----
    def __init__(self, hdf5_path: str, split: str = "train",
                 transform=None):
        self.hdf5_path = hdf5_path
        self.split     = split
        self.transform = transform
        self.samples   = []   # list of (png_path, label_int)
        self.samples_metadata   = []   # list of (src, subject_id)

        self._load_index()

    def _load_index(self):
        """Read all (png_path, label) pairs from HDF5 into memory."""
        group_path = f"mydataset/{self.split}"
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            if group_path not in f:
                raise KeyError(f"Group '{group_path}' not found in HDF5.")
            group   = f[group_path]
            skipped = 0

            for key in group.keys():
                attrs     = dict(group[key].attrs)
                raw_class = attrs.get("class",   b"")
                png_path  = attrs.get("pngPath", b"")
                src = attrs.get("source_id", b"")
                sub = attrs.get("ID", b"")

                if isinstance(raw_class, bytes):
                    raw_class = raw_class.decode("utf-8")
                if isinstance(png_path, bytes):
                    png_path = png_path.decode("utf-8")

                raw_class = raw_class.strip()
                png_path  = png_path.strip()
                src = int(src)

                if raw_class not in HITSDataset.CLASS_TO_IDX or not png_path:
                    skipped += 1
                    continue

                self.samples.append((png_path, HITSDataset.CLASS_TO_IDX[raw_class]))
                self.samples_metadata.append((src, sub))

            if skipped:
                print(f"  [INFO] Skipped {skipped} invalid samples "
                      f"({self.split})")

        # ── Class distribution summary ───────────────────────────────────
        dist  = Counter(lbl for _, lbl in self.samples)
        total = len(self.samples)
        print(f"  [{self.split}] {total} samples loaded")
        for idx, cname in enumerate(HITSDataset.CLASSES):
            n = dist.get(idx, 0)
            print(f"    {cname}: {n} ({n/total*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        png_path, label = self.samples[idx]
        try:
            img = Image.open(png_path).convert("RGB")
        except Exception as e:
            print(f"  [WARNING] Cannot load {png_path}: {e}")
            img = Image.new("RGB", (224, 96), color=0)
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def get_indices_by_src(self, src_id: int):
        """Return a list of indices for samples with the given source ID."""
        return [i for i, (src, _) in enumerate(self.samples_metadata) if src == src_id]
    
    def get_indices_by_sub(self, sub_id: str):
        """Return a list of indices for samples with the given subject ID."""
        return [i for i, (_, sub) in enumerate(self.samples_metadata) if sub == sub_id]


# =============================================================================
# Stratified Val Split helper
# =============================================================================

def _stratified_split(dataset: HITSDataset, val_pct: float = 0.15,
                      seed: int = 42):
    """
    Carve out val_pct of each class from a HITSDataset to form a
    stratified validation set. Returns (train_subset, val_subset).

    Args:
        dataset : HITSDataset (train split)
        val_pct : fraction to use for validation (default 0.15)
        seed    : random seed for reproducibility
    """
    rng = random.Random(seed)

    class_indices = {}
    for i, (_, lbl) in enumerate(dataset.samples):
        class_indices.setdefault(lbl, []).append(i)

    train_idx, val_idx = [], []
    for lbl, idxs in class_indices.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_pct))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return Subset(dataset, train_idx), Subset(dataset, val_idx)

    