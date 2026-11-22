"""
figures.py - classification maps and result plots for MSCM-net (Indian Pines and Salinas).
make_figures(...) loads a trained checkpoint, predicts the whole scene and saves two PNGs.
plot_maps / plot_metrics are pure matplotlib (numpy in, PNG out).
"""
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

DATASETS = {
    "indian_pines": dict(
        title="Indian Pines", cache="indian_pines", default_tag="ip_aug", paper_oa=94.35, bands=(29, 19, 9),
        names=["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
               "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
               "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees", "Stone-Steel-Towers"],
        tags={"Baseline": "indian_pines", "+ Dropout 0.3": "ip_drop03", "+ Weight decay": "ip_wd",
              "+ Augmentation": "ip_aug", "+ Aug + Drop 0.3": "ip_aug_drop03"}),
    "salinas": dict(
        title="Salinas", cache="salinas", default_tag="sal_aug", paper_oa=98.35, bands=(30, 20, 10),
        names=["Broccoli weeds 1", "Broccoli weeds 2", "Fallow", "Fallow rough plow", "Fallow smooth",
               "Stubble", "Celery", "Grapes untrained", "Soil vineyard develop", "Corn senesced weeds",
               "Lettuce romaine 4wk", "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
               "Vineyard untrained", "Vineyard vertical trellis"],
        tags={"Baseline": "sal_base", "+ Augmentation": "sal_aug"}),
}
# index 0 = background (white), 1..16 = classes
CMAP = ListedColormap(["white"] + [plt.cm.tab20(i) for i in range(16)])


def false_color(img, bands):
    rgb = img[:, :, list(bands)].astype(np.float64)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], (2, 98))
        rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb


def plot_maps(img, gt, pred_full, train_idx, test_idx, path, names, title, bands=(29, 19, 9), title_suffix=""):
    """pred_full: (H, W) predicted class 1..16 for every pixel."""
    H, W = gt.shape
    labeled = gt > 0
    pred_masked = np.where(labeled, pred_full, 0)

    status = np.zeros((H, W), dtype=int)                       # 0 bg, 1 train, 2 test correct, 3 test wrong
    status[tuple(train_idx.T)] = 1
    r, c = test_idx[:, 0], test_idx[:, 1]
    status[r, c] = np.where(pred_full[r, c] == gt[r, c], 2, 3)
    err_cmap = ListedColormap(["white", "#9ecae1", "#d9d9d9", "#e31a1c"])
    acc = 100 * (status == 2).sum() / len(test_idx)

    narrow = W / H < 0.8                                        # tall scenes such as Salinas
    fig, ax = plt.subplots(1, 4, figsize=(12, 8.5) if narrow else (19, 6))
    ax[0].imshow(false_color(img, bands)); ax[0].set_title("False-colour composite")
    ax[1].imshow(gt, cmap=CMAP, vmin=0, vmax=16, interpolation="nearest"); ax[1].set_title("Ground truth")
    ax[2].imshow(pred_masked, cmap=CMAP, vmin=0, vmax=16, interpolation="nearest")
    ax[2].set_title("MSCM-net prediction")
    ax[3].imshow(status, cmap=err_cmap, vmin=0, vmax=3, interpolation="nearest")
    ax[3].set_title(f"Test errors  (test OA {acc:.2f}%)")
    for a in ax:
        a.axis("off")
    ax[3].legend(handles=[Patch(color="#9ecae1", label="training pixel"),
                          Patch(color="#d9d9d9", label="test, correct"),
                          Patch(color="#e31a1c", label="test, wrong")],
                 loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, frameon=False)
    fig.legend(handles=[Patch(color=CMAP(i + 1), label=f"{i+1} {n}") for i, n in enumerate(names)],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4 if narrow else 8, fontsize=9, frameon=False)
    fig.suptitle(title + title_suffix, fontsize=14)
    fig.tight_layout(rect=(0, 0.22 if narrow else 0.16, 1, 0.96))
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.show()


def plot_metrics(per_class, R, tags, path, names, paper_oa):
    """Left: per-class accuracy of the shown model. Right: mean +/- std OA over seeds per configuration."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(16)
    ax[0].barh(y, per_class, color=[CMAP(i + 1) for i in range(16)])
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{i+1} {n}" for i, n in enumerate(names)], fontsize=9)
    ax[0].invert_yaxis(); ax[0].set_xlim(0, 105); ax[0].set_xlabel("Test accuracy (%)")
    ax[0].set_title("Per-class accuracy (single seed)")
    for yi, v in zip(y, per_class):
        ax[0].text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=8)

    labels, means, stds = [], [], []
    for label, tag in tags.items():
        fs = sorted(glob.glob(f"{R}/results/{tag}_seed*.npz"))
        if not fs:
            continue
        oa = np.array([float(np.load(f)["OA"]) for f in fs])
        labels.append(f"{label}\n(n={len(oa)})"); means.append(oa.mean()); stds.append(oa.std())
    x = np.arange(len(labels))
    ax[1].bar(x, means, yerr=stds, capsize=5, color="#6baed6")
    ax[1].axhline(paper_oa, ls="--", color="k", lw=1)
    ax[1].text(-0.4, paper_oa + 0.3, f"paper (reported): {paper_oa}", ha="left", fontsize=9)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9)
    top = max(97, paper_oa + 2)
    ax[1].set_ylim(max(0, min(m - s for m, s in zip(means, stds)) - 2), top)
    ax[1].set_ylabel("Overall accuracy (%)  [y-axis truncated]")
    ax[1].set_title("Overall accuracy over seeds (mean ± std)")
    for xi, m, sd in zip(x, means, stds):
        ax[1].text(xi, m + sd + 0.25, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.show()


def make_figures(R, data_path, gt_path, dataset="indian_pines", tag=None, seed=None, in_ch=30, patch=15):
    """Needs torch. Picks the seed whose OA is closest to the mean of `tag`, unless `seed` is given."""
    import torch
    from torch.utils.data import DataLoader
    from hsi_data import build, load_hsi, HSIPatches
    from mscm_net import MSCMNet
    from train import evaluate

    cfg = DATASETS[dataset]
    tag = tag or cfg["default_tag"]
    oas = np.array([float(np.load(f"{R}/results/{tag}_seed{s}.npz")["OA"]) for s in range(10)])
    if seed is None:
        seed = int(np.argmin(np.abs(oas - oas.mean())))        # seed closest to the mean OA: representative
    print(f"using {tag} seed {seed} (OA {oas[seed]:.2f}; mean over seeds {oas.mean():.2f})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds, gt = build(data_path, gt_path, n_components=in_ch, patch=patch, n_train=30,
                                  seed=seed, cache_dir=f"{R}/data/cache", name=cfg["cache"])
    img, _ = load_hsi(data_path, gt_path)
    k = int(gt.max())
    model = MSCMNet(in_ch=in_ch, num_classes=k).to(device)
    model.load_state_dict(torch.load(f"{R}/checkpoints/{tag}_seed{seed}.pt", map_location=device))
    model.eval()

    H, W = gt.shape
    all_idx = np.argwhere(np.ones((H, W), dtype=bool))
    full_ds = HSIPatches(train_ds.x.numpy(), gt, all_idx, patch)   # same padded PCA image, every pixel
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(full_ds, batch_size=256):
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    pred_full = (np.concatenate(preds) + 1).reshape(H, W)

    res = evaluate(model, DataLoader(test_ds, batch_size=256), k, device)
    print(f"test OA {res['OA']:.2f}  AA {res['AA']:.2f}  Kappa {res['Kappa']:.2f}")

    out = f"{R}/results/figures"; os.makedirs(out, exist_ok=True)
    plot_maps(img, gt, pred_full, train_ds.idx, test_ds.idx, f"{out}/{dataset}_maps.png",
              cfg["names"], cfg["title"], cfg["bands"], title_suffix=f" — MSCM-net (seed {seed})")
    plot_metrics(res["per_class"], R, cfg["tags"], f"{out}/{dataset}_metrics.png", cfg["names"], cfg["paper_oa"])
    print("saved to", out)
    return res