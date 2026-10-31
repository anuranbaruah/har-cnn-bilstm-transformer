"""Shared preprocessing pipeline for UCI HAR (used by every model).

Steps:
1. Load the 9 raw inertial signal files (128 timesteps per window) for train and test.
2. Stack into tensors of shape (N, 128, 9).
3. Subject-wise split: the official 9 test subjects stay as the test set; 4 of the 21
   training subjects are held out as a validation set, so no subject appears in two splits.
4. Per-channel z-score normalisation, with mean/std computed on the training split only.
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
CLASSES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
VAL_SUBJECTS = [1, 11, 21, 27]  # 4 of the 21 official training subjects


def _load_split(root, split):
    sig_dir = os.path.join(root, split, "Inertial Signals")
    X = np.stack([np.loadtxt(os.path.join(sig_dir, f"{s}_{split}.txt"), dtype=np.float32)
                  for s in SIGNALS], axis=-1)                       # (N, 128, 9)
    y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt"), dtype=np.int64) - 1  # 0..5
    subj = np.loadtxt(os.path.join(root, split, f"subject_{split}.txt"), dtype=np.int64)
    return X, y, subj


def load_har(root):
    Xtr, ytr, str_ = _load_split(root, "train")
    Xte, yte, _ = _load_split(root, "test")

    val_mask = np.isin(str_, VAL_SUBJECTS)
    Xva, yva = Xtr[val_mask], ytr[val_mask]
    Xtr, ytr = Xtr[~val_mask], ytr[~val_mask]

    mean = Xtr.reshape(-1, 9).mean(axis=0)
    std = Xtr.reshape(-1, 9).std(axis=0) + 1e-8
    norm = lambda X: (X - mean) / std
    return {"train": (norm(Xtr), ytr), "val": (norm(Xva), yva), "test": (norm(Xte), yte),
            "mean": mean, "std": std}


def make_loaders(data, batch_size=64, seed=42):
    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in ["train", "val", "test"]:
        X, y = data[split]
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), generator=g)
    return loaders
