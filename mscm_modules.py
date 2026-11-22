"""
mscm_modules.py - Stage 1 of MSCM-net: spectral compression stem + MCSE module.
Shapes assume input (B, C=30, 15, 15).
"""
import torch
import torch.nn as nn


class Stem(nn.Module):
    """Paper Sec 2.1: 3x3 conv, C -> 64 channels. padding=0 explains 15x15 -> 13x13."""
    def __init__(self, in_ch=30, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):          # (B, C, 15, 15) -> (B, 64, 13, 13)
        return self.net(x)


class DWSeparable(nn.Module):
    """Depthwise k x k conv (groups=channels) -> 1x1 pointwise -> BN -> ReLU. Eq. 2-5."""
    def __init__(self, ch, k):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch)  # keeps 13x13
        self.pw = nn.Conv2d(ch, ch, 1)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class SE(nn.Module):
    """Squeeze-and-Excitation. Eq. 7-9. Reduction ratio is NOT given in the paper (16 assumed)."""
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x):
        z = x.mean(dim=(2, 3))                                   # squeeze: (B, ch)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))     # excite:  (B, ch)
        return x * s[:, :, None, None]                           # reweight channels


class MCSE(nn.Module):
    """
    Three parallel branches (1x1, 3x3 DW-sep, 5x5 DW-sep) -> concat (3*64=192)
    -> SE -> BN -> ReLU -> 1x1 conv to `out_dim` (128).
    The final 1x1 projection is NOT in the paper (it never explains 192 -> 128);
    this is our design decision.
    """
    def __init__(self, in_ch=64, out_dim=128, se_r=16):
        super().__init__()
        self.b1 = nn.Conv2d(in_ch, in_ch, 1)          # Eq. 1: plain channel mixing
        self.b3 = DWSeparable(in_ch, 3)
        self.b5 = DWSeparable(in_ch, 5)
        cat_ch = 3 * in_ch
        self.se = SE(cat_ch, se_r)
        self.bn = nn.BatchNorm2d(cat_ch)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(cat_ch, out_dim, 1)     # design decision (see docstring)

    def forward(self, x):          # (B, 64, 13, 13) -> (B, 128, 13, 13)
        x = torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)   # (B, 192, 13, 13)
        x = self.act(self.bn(self.se(x)))
        return self.proj(x)


if __name__ == "__main__":
    x = torch.randn(4, 30, 15, 15)
    stem, mcse = Stem(30, 64), MCSE(64, 128)
    h = stem(x);  print("stem :", tuple(h.shape))    # (4, 64, 13, 13)
    o = mcse(h);  print("mcse :", tuple(o.shape))    # (4, 128, 13, 13)
    n = sum(p.numel() for m in (stem, mcse) for p in m.parameters())
    print("params (stem+MCSE):", n)
