"""
mscm_net.py - Mamba blocks (pure PyTorch), dual-branch head, and the full MSCM-net.
Needs mscm_modules.py (Stem, MCSE) in the same folder.
Input: (B, C=30, 15, 15) -> logits (B, num_classes)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mscm_modules import Stem, MCSE


class SelectiveSSM(nn.Module):
    """
    Selective state space layer (Eq. 15-21), written as a plain loop over the sequence.
    dt, B, C depend on the input (that is the 'selective' part); A is a learned matrix.
    """
    def __init__(self, d_inner, d_state=16, dt_rank=8):
        super().__init__()
        self.d_inner, self.d_state, self.dt_rank = d_inner, d_state, dt_rank
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        # A initialised as 1..N for every channel, stored as log so that A = -exp(A_log) < 0 (stable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))          # skip connection term (D*x in Eq. 16)
        self.use_loop = True                                # True = sequential scan (default). False = parallel scan, needs ~20+ GB when training

    def forward(self, x):                                   # x: (B, L, Di)
        Bsz, L, Di = x.shape
        A = -torch.exp(self.A_log)                          # (Di, N)
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                   # (B, L, Di)   Eq. 17

        dA = torch.exp(dt.unsqueeze(-1) * A)                # (B, L, Di, N)  Eq. 20
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)   # (B, L, Di, N)  (Eq. 21, Euler form of B-bar)

        h = self._scan_loop(dA, dBx) if self.use_loop else self._scan_parallel(dA, dBx)   # (B, L, Di, N)
        y = (h * Cm.unsqueeze(2)).sum(-1)                   # Eq. 16: y_t = C_t h_t   (B, L, Di)
        return y + x * self.D

    @staticmethod
    def _scan_loop(a, b):
        """Slow reference: h_t = a_t * h_{t-1} + b_t, one token at a time (Eq. 15)."""
        # unbind() instead of a[:, t]: indexing makes autograd allocate a full-size zero
        # gradient tensor at every step, which was the main cost of training.
        h = torch.zeros_like(b[:, 0])
        out = []
        for a_t, b_t in zip(a.unbind(1), b.unbind(1)):
            h = a_t * h + b_t
            out.append(h)
        return torch.stack(out, dim=1)

    @staticmethod
    def _scan_parallel(a, b):
        """
        Same recurrence in log2(L) vectorised steps (Hillis-Steele doubling scan).
        Each position holds (a, b) for a window of tokens; windows are merged with the one
        `shift` positions earlier: b <- b + a * b_prev,  a <- a * a_prev.
        All a values lie in (0, 1], so nothing overflows.
        """
        L = a.shape[1]
        shift = 1
        while shift < L:
            pad = (0, 0, 0, 0, shift, 0)                    # pad the token axis at the front
            b_prev = F.pad(b, pad, value=0.0)[:, :L]
            a_prev = F.pad(a, pad, value=1.0)[:, :L]
            b = b + a * b_prev
            a = a * a_prev
            shift *= 2
        return b


class MambaBlock(nn.Module):
    """LayerNorm -> in_proj -> (causal depthwise conv1d -> SiLU -> SSM) * SiLU(gate) -> out_proj -> + residual."""
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.ssm = SelectiveSSM(d_inner, d_state, dt_rank=math.ceil(d_model / 16))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):                                   # (B, L, D)
        L = x.shape[1]
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)        # each (B, L, Di)
        u = self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2)   # trim -> causal, same length
        u = self.ssm(F.silu(u))
        return x + self.out_proj(u * F.silu(gate))


class MSCMNet(nn.Module):
    def __init__(self, in_ch=30, num_classes=16, dim=128, depth=4, grid=13,
                 d_state=16, expand=2, dropout=0.1, pos_alpha=1.0):
        super().__init__()
        self.stem = Stem(in_ch, 64)
        self.mcse = MCSE(64, dim)
        self.orig_proj = nn.Linear(64, dim)                              # Eq. 12
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))        # E_pos (Eq. 14)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.alpha = pos_alpha
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state, expand=expand) for _ in range(depth)])
        self.center_fc = nn.Linear(dim, dim)                             # Eq. 24
        self.head = nn.Sequential(nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(inplace=True),
                                  nn.Linear(dim, num_classes))

    def forward(self, x):                                   # (B, C, 15, 15)
        h = self.stem(x)                                    # (B, 64, 13, 13)
        m = self.mcse(h)                                    # (B, 128, 13, 13)
        seq_m = m.flatten(2).transpose(1, 2)                # (B, 169, 128)   Eq. 10
        seq_o = h.flatten(2).transpose(1, 2)                # (B, 169, 64)    Eq. 11
        o = F.relu(self.drop(self.orig_proj(seq_o)))        # Eq. 12
        z = self.drop(o + seq_m)                            # Eq. 13 (residual before Mamba)
        z = z + self.pos * self.alpha                       # Eq. 14
        for blk in self.blocks:
            z = blk(z)                                      # (B, 169, 128)
        L = z.shape[1]
        center = F.relu(self.drop(self.center_fc(z[:, L // 2, :])))      # Eq. 22-24, index 84
        glob = z.mean(dim=1)                                             # Eq. 25
        return self.head(center + glob)                                  # Eq. 26 -> classifier


if __name__ == "__main__":
    model = MSCMNet(in_ch=30, num_classes=16)
    x = torch.randn(4, 30, 15, 15)
    print("logits:", tuple(model(x).shape))                 # (4, 16)
    print("params:", sum(p.numel() for p in model.parameters()))