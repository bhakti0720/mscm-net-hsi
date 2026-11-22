"""
hsi_data.py - preprocessing for MSCM-net reproduction.
Pipeline: load .mat -> per-band standardize -> PCA -> zero-pad -> per-class split -> patch Dataset
"""
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA


# ---------- 1. Loading ----------
def _first_array(path):
    """Load the first non-metadata array from a .mat file (handles v7.3/HDF5 too)."""
    try:
        d = sio.loadmat(path)
        keys = [k for k in d if not k.startswith("__")]
        return d[keys[0]]
    except NotImplementedError:  # MATLAB v7.3 -> HDF5
        import h5py
        with h5py.File(path, "r") as f:
            k = list(f.keys())[0]
            arr = np.array(f[k])
        return arr.T  # h5py returns reversed axes


def load_hsi(data_path, gt_path):
    img = _first_array(data_path).astype(np.float32)
    gt = _first_array(gt_path).astype(np.int64)
    assert img.ndim == 3 and gt.ndim == 2, f"unexpected shapes {img.shape} {gt.shape}"

    # Some uploads store the cube channels-first (B, H, W). Convert to (H, W, B).
    if img.shape[:2] != gt.shape and img.shape[1:] == gt.shape:
        img = np.moveaxis(img, 0, -1)

    assert img.shape[:2] == gt.shape, f"image/gt mismatch {img.shape} vs {gt.shape}"
    return img, gt


# ---------- 2. Normalize + PCA ----------
def standardize_pca(img, n_components=30, seed=0):
    H, W, B = img.shape
    x = img.reshape(-1, B)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)            # per-band z-score
    x = PCA(n_components=n_components, random_state=seed).fit_transform(x)
    x = x.reshape(H, W, n_components).astype(np.float32)
    # rescale each component to zero mean / unit std so scales are comparable
    x = (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-8)
    return x


# ---------- 3. Pad ----------
def pad_image(x, patch=15):
    r = patch // 2
    return np.pad(x, ((r, r), (r, r), (0, 0)), mode="constant")


# ---------- 4. Split ----------
def split_per_class(gt, n_train=30, seed=0):
    """
    n_train labeled pixels per class for training, rest for testing.
    Classes with <= n_train pixels use half for training (matches paper Table 1:
    Grass-pasture-mowed 14/14, Oats 10/10).
    Returns (train_idx, test_idx) as arrays of (row, col).
    """
    rng = np.random.RandomState(seed)
    train, test = [], []
    for c in np.unique(gt):
        if c == 0:
            continue
        pos = np.argwhere(gt == c)
        rng.shuffle(pos)
        n = len(pos)
        k = n // 2 if n <= n_train else n_train
        train.append(pos[:k])
        test.append(pos[k:])
    train, test = np.concatenate(train), np.concatenate(test)
    return train, test


# ---------- 5. Dataset (patches built on the fly) ----------
class HSIPatches(Dataset):
    def __init__(self, padded, gt, idx, patch=15):
        self.x = torch.from_numpy(padded)                  # (H+2r, W+2r, C)
        self.idx = idx
        self.labels = torch.from_numpy(gt[idx[:, 0], idx[:, 1]] - 1)  # 0..K-1
        self.p = patch

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r, c = self.idx[i]                                 # padded coords = r..r+p
        patch = self.x[r:r + self.p, c:c + self.p, :]      # (p, p, C)
        return patch.permute(2, 0, 1).contiguous(), self.labels[i]


# ---------- 6. One-call builder with caching ----------
def build(data_path, gt_path, n_components=30, patch=15, n_train=30, seed=0,
          cache_dir=None, name="ds"):
    cache = os.path.join(cache_dir, f"{name}_pca{n_components}.npz") if cache_dir else None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        X, gt = z["X"], z["gt"]
    else:
        img, gt = load_hsi(data_path, gt_path)
        X = standardize_pca(img, n_components, seed)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache, X=X, gt=gt)
    padded = pad_image(X, patch)
    tr, te = split_per_class(gt, n_train, seed)
    return HSIPatches(padded, gt, tr, patch), HSIPatches(padded, gt, te, patch), gt


def report(gt, train_ds, test_ds):
    k = gt.max()
    tr = np.bincount(train_ds.labels.numpy(), minlength=k)
    te = np.bincount(test_ds.labels.numpy(), minlength=k)
    print(f"classes={k}  train={tr.sum()}  test={te.sum()}")
    for i in range(k):
        print(f"  class {i+1:2d}: train {tr[i]:4d}  test {te[i]:6d}")
