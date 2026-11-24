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

import argparse
import os
import sys
from collections import Counter

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

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
    BASE_HITS_TASK_SRC = {
        'HITSsrc':{
            'task_list': [(6,), (5,), (9,), (1, 3,), (2,), (4, 8,)], #, (7, 10,)]
            'color_dict': {
                (6,): "tab:blue",
                (5,): "tab:orange",
                (9,): "tab:green",
                (1, 3, 11): "tab:red",
                (2,): "tab:purple",
                (4, 8): "tab:brown",
            },
            'explication': '''
            This is individual hospital sources, or grouping of similar hospitalsources, to maximize the number of tasks from 
            the initial 51 patients, using same train and test than in MICCAI26-CLiMeM dataset.
                6:                    mainly A/GE, some FOP
                5:                    1969 total, all 3 classes
                9:                    609 total, mainly A/SE, stenosis only
                TAVI group 1-3:    396 total, mainly A/GE, TAVI
                    1:  GE only
                    3:  mainly GE
                    11: A only
                2:                     mainly A/SE
                stenosis group 4-8:    mainly A/SE, stenosis
                    4:  mainly A/SE, stenosis
                    8:  mainly SE/A, stenosis only
                remaining group 7-10:  2 sub,  139 total, mainly A/SE, stenosis only
                    7:  GE only
                    10: mainly A/GE, AFA
            '''
        },
        'HITSsrcAB':{
            'task_list': [(5, 6, 9,), (1, 2, 3, 7, 8)],
            'color_dict': {
                (5, 6, 9, 11): "tab:blue",
                (1, 2, 3, 7, 8): "tab:orange",
            },
            'explication': '''
            This is a grouping of the HITS task sources into two main groups: A and B, same as in MICCAI26-CLiMeM work.
            Group A includes sources 5, 6, and 9.
            Group B includes sources 1, 2, 3, 7, and 8.
            4 and 10 were used as external test sets.
            '''
        },
        'HITSsrcA-Bsplit':{
            'task_list': [(5, 6, 9,), (1,), (2,), (3,), (7,), (8,)],
            'color_dict': {
                (5, 6, 9, 11): "tab:blue",
                (1,): "tab:green",
                (2,): "tab:red",
                (3,): "tab:purple",
                (7,): "tab:brown",
                (8,): "tab:olive",
            },
            'explication': '''
            This is a grouping of the HITS task sources into one big task A, and divided task B into smaller tasks.
            '''
        },
        'HITSsrcB-Asplit':{
            'task_list': [(1, 2, 3, 7, 8,), (5,), (6,), (9,)],
            'color_dict': {
                (1, 2, 3, 7, 8,): "tab:orange",
                (5,): "tab:green",
                (6,): "tab:red",
                (9,): "tab:purple",
            },
            'explication': '''
            This is a grouping of the HITS task sources into one big task B, and divided task A into smaller tasks.
            '''
        },
        'HITSsrc-split-all':{
            'task_list': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)],
            'color_dict': {
                (1,): "tab:blue",
                (2,): "tab:orange",
                (3,): "tab:green",
                (4,): "tab:red",
                (5,): "tab:purple",
                (6,): "tab:brown",
                (7,): "tab:pink",
                (8,): "tab:gray",
                (9,): "tab:olive",
                (10,): "tab:cyan",
            },
            'explication': '''
            This is a grouping of the HITS task sources into individual tasks for each source.
            '''
        }
    }
    
    @classmethod
    def get_task_colors(cls, experiment_name):
        """Return plotting colors aligned with the current HITS task order."""
        assert experiment_name in cls.BASE_HITS_TASK_SRC.keys(), f"Experiment name '{experiment_name}' not found in BASE_HITS_TASK_SRC"
        color_dict = cls.BASE_HITS_TASK_SRC[experiment_name].get('color_dict', None)
        return [color_dict[task] for task in cls.BASE_HITS_TASK_SRC[experiment_name].get('task_list', [])]

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


def _count_labels(dataset, indices):
    """Return class-count dict for selected indices of a HITSDataset."""
    counts = Counter(dataset.samples[i][1] for i in indices)
    return {HITSDataset.CLASSES[i]: counts.get(i, 0) for i in range(len(HITSDataset.CLASSES))}


def _print_task_class_breakdown(split_name, dataset, task_groups):
    """Print per-task class composition for the provided split."""
    print(f"\n=== {split_name.upper()} TASK BREAKDOWN ===")
    for task_id, src_group in enumerate(task_groups, start=1):
        indices = []
        for src in src_group:
            indices.extend(dataset.get_indices_by_src(src))

        class_counts = _count_labels(dataset, indices)
        total = sum(class_counts.values())

        print(f"Task {task_id} | sources={src_group} | total={total}")
        for class_name in HITSDataset.CLASSES:
            count = class_counts[class_name]
            pct = (100.0 * count / total) if total > 0 else 0.0
            print(f"  {class_name}: {count} ({pct:.1f}%)")


def _plot_task_charts(split_name, dataset, task_groups, out_dir, d_name='hits', plot_type='pie'):
    """Plot and save one pie chart per task for class composition."""
    assert plot_type in ["pie", "bar"], f"Plot type should be 'pie' or 'bin, not {plot_type}"
    class_colors = {
        "A": "#F2422B",
        "ES": "#739DCA",
        "EG": "#3AAA2C",
    }
    n_tasks = len(task_groups)
    n_cols = 5
    n_rows = int(np.ceil(n_tasks / n_cols))
    labels = HITSDataset.CLASSES
    
    if plot_type == "pie":
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).reshape(-1)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(max(10, 1.2 * n_tasks), 6))
        bottoms = np.zeros(n_tasks)
        x = np.arange(n_tasks)


    class_totals = {label: [] for label in labels}
    task_totals = []
    for i, src_group in enumerate(task_groups):
        indices = []
        for src in src_group:
            indices.extend(dataset.get_indices_by_src(src))

        class_counts = _count_labels(dataset, indices)
        total = sum(class_counts.values())
        task_totals.append(total)
        for label in labels:
            class_totals[label].append(class_counts[label])

        if plot_type == 'pie':
            ax = axes[i]
            if total == 0:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center", fontsize=12)
                ax.set_title(f"Task {i+1} | src={src_group}")
                ax.axis("off")
                continue

            pie_colors = [class_colors[label] for label in labels]
            ax.pie(
                [class_totals[label][i] for label in labels],
                labels=labels,
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
            )
            ax.set_title(f"Task {i+1} | src={src_group} | n={total}")
            ax.axis("equal")

            for j in range(n_tasks, len(axes)):
                axes[j].axis("off")
    if plot_type == "bar":
        for label in labels:
            values = np.array(class_totals[label])
            ax.bar(
                x,
                values,
                bottom=bottoms,
                label=label,
                color=class_colors[label],
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms += values

        ax.set_xticks(x)
        ax.set_xticklabels([f"Task {i + 1}" for i in x])
        ax.set_xlabel("Task")
        ax.set_ylabel("Number of HITS")
        ax.set_title(f"HITS {split_name.capitalize()} Class Distribution per Task")
        ax.legend(title="Class")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        # Show task totals above bars for easier reading.
        y_pad = max(1.0, 0.015 * max(task_totals) if task_totals else 1.0)
        for i, total in enumerate(task_totals):
            if total > 0:
                ax.text(i, total + y_pad, str(total), ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
    else:
        fig.suptitle(f"HITS {split_name.capitalize()} Class Distribution per Task", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{d_name}_{split_name}_class_{plot_type}s.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_type} charts to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Inspect HITS source-group tasks and print class-wise details per task. "
            "Task formation matches data/load.py:get_multitask_experiment for HITSsrc."
        )
    )
    parser.add_argument(
        "--hdf5",
        default="./store/datasets/data_5k-srcCL-subsplit.hdf5",
        help="Path to HITS HDF5 file (default: ./store/datasets/data_5k-srcCL-subsplit.hdf5)",
    )
    parser.add_argument(
        "--base_tasks",
        default="HITSsrc",
        help="Base name of the HITS dataset (default: HITSsrc)",
        choices=HITSDataset.BASE_HITS_TASK_SRC.keys()
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate task charts for train and test splits.",
    )
    parser.add_argument(
        "--plot-type",
        default="pie",
        choices=["pie", "bar"],
        help=(
            "Chart type to generate when --plot is enabled: "
            "'pie' for per-task pie charts, 'bar' for stacked bars per task."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        default="./store/plots",
        help="Output directory for generated charts (default: ./store/plots)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.hdf5):
        raise FileNotFoundError(
            f"HDF5 file not found: {args.hdf5}. "
            "Pass --hdf5 with the correct path to data_5k-srcCL-subsplit.hdf5"
        )
    assert args.base_tasks in HITSDataset.BASE_HITS_TASK_SRC.keys(), f"For HITS inspection, please precise base tasks (one of: {list(HITSDataset.BASE_HITS_TASK_SRC.keys())})"
    task_groups = HITSDataset.BASE_HITS_TASK_SRC[args.base_tasks]['task_list']

    print(f"Using HDF5: {args.hdf5}")
    print(f"Task groups (HITS_TASK_SRC): {task_groups}")

    train_set = HITSDataset(hdf5_path=args.hdf5, split="train", transform=None)
    test_set = HITSDataset(hdf5_path=args.hdf5, split="test", transform=None)

    _print_task_class_breakdown("train", train_set, task_groups)
    _print_task_class_breakdown("test", test_set, task_groups)

    if args.plot:
        _plot_task_charts("train", train_set, task_groups, args.plot_dir, d_name=args.base_tasks, plot_type=args.plot_type)
        _plot_task_charts("test", test_set, task_groups, args.plot_dir, d_name=args.base_tasks, plot_type=args.plot_type)

