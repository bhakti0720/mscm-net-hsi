"""
train.py - training + evaluation for MSCM-net.
Metrics: OA, AA, Kappa (x100) and per-class accuracy, as in the paper's Tables 2-4.
Settings from paper Sec 3.2: Adam, lr 1e-3, batch 64, 100 epochs.
"""
import os, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hsi_data import build
from mscm_net import MSCMNet


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def random_dihedral(x):
    """Random flip + 90-degree rotation per sample on (B, C, P, P) patches.
    The centre pixel stays the centre, so the label stays valid."""
    out = []
    for p in x:
        p = torch.rot90(p, int(torch.randint(0, 4, (1,))), dims=(1, 2))
        if torch.rand(1).item() < 0.5:
            p = torch.flip(p, dims=(2,))
        out.append(p)
    return torch.stack(out)


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)   # rows = true, cols = predicted
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        np.add.at(cm, (y.numpy(), pred), 1)
    n = cm.sum()
    per_class = np.diag(cm) / np.maximum(cm.sum(1), 1)
    oa = np.trace(cm) / n
    aa = per_class.mean()
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    kappa = (oa - pe) / (1 - pe)
    return dict(OA=100 * oa, AA=100 * aa, Kappa=100 * kappa, per_class=100 * per_class, cm=cm)


def fit(model, train_ds, test_ds, num_classes, epochs=100, lr=1e-3, bs=64,
        eval_every=0, device="cuda", ckpt_path=None, weight_decay=0.0, augment=False):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    # BatchNorm1d in the head fails on a batch of size 1
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              drop_last=(len(train_ds) % bs == 1))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if augment:
                x = random_dihedral(x)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
        if ep == 1 or ep % 10 == 0:
            msg = f"ep {ep:3d}  loss {loss_sum/tot:.4f}  train_acc {100*correct/tot:.1f}%  [{time.time()-t0:.0f}s]"
            if eval_every and ep % eval_every == 0:      # monitoring only, NOT used to pick a model
                msg += f"  test OA {evaluate(model, test_loader, num_classes, device)['OA']:.2f}"
            print(msg)

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    # Report the FINAL-epoch model. Picking the best epoch on the test set would leak test labels.
    return evaluate(model, test_loader, num_classes, device)


def run(data_path, gt_path, name, in_ch=30, patch=15, n_train=30, seed=0, epochs=100,
        cache_dir=None, ckpt_dir=None, weight_decay=0.0, augment=False, tag=None, **model_kw):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds, gt = build(data_path, gt_path, n_components=in_ch, patch=patch,
                                  n_train=n_train, seed=seed, cache_dir=cache_dir, name=name)
    k = int(gt.max())
    model = MSCMNet(in_ch=in_ch, num_classes=k, **model_kw)
    print(f"{name} seed={seed} device={device} classes={k} train={len(train_ds)} test={len(test_ds)} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    ckpt = os.path.join(ckpt_dir, f"{tag or name}_seed{seed}.pt") if ckpt_dir else None
    res = fit(model, train_ds, test_ds, k, epochs=epochs, device=device, ckpt_path=ckpt,
              weight_decay=weight_decay, augment=augment)
    print(f"RESULT {name} seed={seed}: OA {res['OA']:.2f}  AA {res['AA']:.2f}  Kappa {res['Kappa']:.2f}")
    return res