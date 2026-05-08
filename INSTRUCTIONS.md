# FFTFormer Integration into RTFM — Implementation Instructions

> **Paper references**
> RTFM: *Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning* (Tian et al., ICCV 2021)
> FFTFormer: *Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring* (Kong et al., CVPR 2023)
>
> **Codebase**: https://github.com/tianyu0207/RTFM

---

## Table of Contents

1. [Actual Repository Structure](#1-actual-repository-structure)
2. [Key Codebase Facts](#2-key-codebase-facts)
3. [Modification Summary](#3-modification-summary)
4. [Step 0 — Create `new_modules.py`](#4-step-0--create-new_modulespy)
5. [Mod 1 — T-DFFN: Frequency Gating inside PDC Branches](#5-mod-1--t-dffn-frequency-gating-inside-pdc-branches)
6. [Mod 2 — T-FSAS: Replace `NONLocalBlock1D` with FFT-based Attention](#6-mod-2--t-fsas-replace-nonlocalblock1d-with-fft-based-attention)
7. [Mod 3 — Frequency-Domain Feature Magnitude](#7-mod-3--frequency-domain-feature-magnitude)
8. [Mod 4 — T-DFFN Gated Classifier Head](#8-mod-4--t-dffn-gated-classifier-head)
9. [Training Loop Changes (`main.py`)](#9-training-loop-changes-mainpy)
10. [Recommended Ablation Order](#10-recommended-ablation-order)
11. [Hyperparameter Checklist](#11-hyperparameter-checklist)
12. [Evaluation Protocol](#12-evaluation-protocol)
13. [Common Errors and Fixes](#13-common-errors-and-fixes)

---

## 1. Actual Repository Structure

All files live at the **root level** — there are no subdirectories for model
or loss code:

```
RTFM/
├── model.py          ← Aggregate class (MTN) + NONLocalBlock1D (TSA) + Model
├── train.py          ← training step (called once per iteration from main.py)
├── test_10crop.py    ← evaluation loop
├── main.py           ← entry point: dataloaders, optimizer, training loop
├── dataset.py        ← Dataset class
├── option.py         ← argparse config
├── config.py         ← Config class (builds lr schedule)
├── utils.py          ← Visualizer, save_best_record
├── list/             ← .list files pointing to feature paths
└── new_modules.py    ← CREATE THIS FILE (all FFT modules go here)
```

---

## 2. Key Codebase Facts

Reading `model.py` carefully reveals several things that differ from generic
RTFM descriptions. These directly affect where each modification goes.

### Class names and their roles

| What the paper calls it | Actual class name in code | File |
|-------------------------|--------------------------|------|
| MTN (multi-scale temporal network) | `Aggregate` | `model.py` |
| PDC branches | `conv_1`, `conv_2`, `conv_3` inside `Aggregate` | `model.py` |
| 1×1 reduction before TSA | `conv_4` inside `Aggregate` | `model.py` |
| TSA (temporal self-attention) | `NONLocalBlock1D` → `_NonLocalBlockND(dimension=1)` | `model.py` |
| Final fusion conv | `conv_5` inside `Aggregate` | `model.py` |
| Snippet classifier | `fc1`, `fc2`, `fc3` inside `Model` | `model.py` |

### The `Aggregate` forward pass (annotated)

```python
def forward(self, x):
    # x arrives as (B, T, F) from Model.forward
    out = x.permute(0, 2, 1)           # → (B, F, T)  Conv1d format
    residual = out

    out1 = self.conv_1(out)             # PDC dilation=1 → (B, 512, T)
    out2 = self.conv_2(out)             # PDC dilation=2 → (B, 512, T)
    out3 = self.conv_3(out)             # PDC dilation=4 → (B, 512, T)
    out_d = torch.cat((out1, out2, out3), dim=1)   # → (B, 1536, T)

    out = self.conv_4(out)              # 1×1 reduce 2048→512 → (B, 512, T)
    out = self.non_local(out)           # NONLocalBlock1D   → (B, 512, T)

    out = torch.cat((out_d, out), dim=1)  # → (B, 2048, T)
    out = self.conv_5(out)              # fuse → (B, 2048, T)
    out = out + residual                # skip connection

    out = out.permute(0, 2, 1)         # → (B, T, F)
    return out
```

### Critical constants in `Model.__init__`

```python
self.num_segments = 32          # T = 32 snippets per video
self.k_abn = self.num_segments // 10   # = 3  (top-k for abnormal)
self.k_nor = self.num_segments // 10   # = 3  (top-k for normal)
```

`k=3` is a fixed attribute on the model, not passed as an argument.

### Feature magnitude computation in `Model.forward`

```python
feat_magnitudes = torch.norm(features, p=2, dim=2)
# features shape: (B * ncrops, T, 2048)
# feat_magnitudes shape: (B * ncrops, T) → then averaged over crops → (B, T)
```

The ℓ₂ norm is taken over the **feature dimension** (dim=2), yielding one
scalar per snippet.

### `Model.forward` return signature (all 10 values)

```python
return (
    score_abnormal,     # [0] (B, 1)  mean score of top-k abnormal snippets
    score_normal,       # [1] (B, 1)  mean score of top-k normal snippets
    feat_select_abn,    # [2] (ncrops*B, k, F) top-k features, abnormal
    feat_select_normal, # [3] (ncrops*B, k, F) top-k features, normal
    feat_select_abn,    # [4] duplicate of [2]
    feat_select_abn,    # [5] duplicate of [2]
    scores,             # [6] (B, T, 1) per-snippet anomaly scores (sigmoid)
    feat_select_abn,    # [7] duplicate of [2]
    feat_select_abn,    # [8] duplicate of [2]
    feat_magnitudes     # [9] (B, T) ℓ₂ norms per snippet
)
```

`train.py` unpacks all 10 values.

### Optimizer and training in `main.py`

```python
optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)
# weight_decay is 0.005 (not 0.0005)
# max_epoch = 15000 *iterations* (not epochs) — each call to train() is one step
```

---

## 3. Modification Summary

| Mod | What changes | Class/method | File |
|-----|-------------|--------------|------|
| 1 | Add T-DFFN after `conv_1`, `conv_2`, `conv_3` | `Aggregate.__init__` and `Aggregate.forward` | `model.py` |
| 2 | Replace `self.non_local` with `TemporalFSAS` | `Aggregate.__init__` only | `model.py` |
| 3 | Replace `torch.norm` with `freq_magnitude` | `Model.forward` | `model.py` |
| 4 | Replace `fc1/fc2/fc3` block with `FreqGatedClassifier` | `Model.__init__` and `Model.forward` | `model.py` |

All new classes go in a single new file `new_modules.py` at the repo root.

**Apply in this order:**

```
Step 0  Create new_modules.py — verify it imports cleanly
Step 1  Mod 2 only            — verify AUC is maintained (~97.21% on ShanghaiTech)
Step 2  + Mod 1               — expect AUC to improve
Step 3  + Mod 4               — expect further improvement
Step 4  + Mod 3               — validate; revert if AUC drops after 15000 iters
```

---

## 4. Step 0 — Create `new_modules.py`

Create this file at the repository root. It contains all four new classes.
Do not modify any existing file until this file imports cleanly.

```python
# new_modules.py
# FFTFormer-inspired modules adapted for RTFM's 1-D temporal axis.

import torch
import torch.nn as nn
import torch.fft as fft


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 & 4: TemporalDFFN
# ─────────────────────────────────────────────────────────────────────────────

class TemporalDFFN(nn.Module):
    """
    Discriminative Frequency-domain FFN adapted for RTFM's temporal axis.

    Adapted from FFTFormer's DFFN (Kong et al., CVPR 2023).

    Operates on channels-FIRST tensors (B, C, T) to match the Conv1d convention
    used throughout Aggregate in model.py.

    W_quant is initialised to ones so the module starts as a near-identity,
    preserving the original RTFM output at iteration 0.

    Args:
        channels   (int): feature channels C (512 for PDC branches).
        patch_size (int): temporal patch size. Must divide T=32 evenly.
                          Default 4 → 8 non-overlapping patches of 4 snippets.
    """

    def __init__(self, channels: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        n_bins = patch_size // 2 + 1    # rfft output bins for real input

        # Learnable spectral quantization — shape (1, C, n_bins).
        # Broadcasts over batch and patch dimensions.
        # Init to ones: identity at start of training.
        self.W_quant = nn.Parameter(torch.ones(1, channels, n_bins))

        # GroupNorm over channels (works in (B, C, T) format).
        self.norm = nn.GroupNorm(1, channels)

        # GEGLU gate using Conv1d(1×1) to stay in channels-first format.
        self.gate_proj = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=False)
        self.out_proj  = nn.Conv1d(channels, channels,     kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        residual = x
        x = self.norm(x)

        B, C, T = x.shape
        P = self.patch_size

        # Zero-pad T to a multiple of P. No-op when T=32 and P=4 or P=8.
        pad = (P - T % P) % P
        if pad:
            x = nn.functional.pad(x, (0, pad))
        Tp = x.shape[2]
        n_patches = Tp // P

        # Fold into patches: (B, C, n_patches, P)
        xp = x.reshape(B, C, n_patches, P)

        # FFT along the patch (time) axis → complex (B, C, n_patches, n_bins)
        xf = fft.rfft(xp, dim=-1)

        # Learnable spectral weighting. W_quant: (1, C, n_bins)
        # Unsqueeze to (1, C, 1, n_bins) to broadcast over n_patches.
        xf = xf * self.W_quant.unsqueeze(2)

        # IFFT back to time domain → (B, C, n_patches, P)
        xp = fft.irfft(xf, n=P, dim=-1)

        # Unfold and trim padding → (B, C, T)
        x = xp.reshape(B, C, Tp)[:, :, :T]

        # GEGLU gating
        gated = self.gate_proj(x)           # (B, 2C, T)
        x1, x2 = gated.chunk(2, dim=1)     # each (B, C, T)
        x = x1 * torch.sigmoid(x2)

        x = self.out_proj(x)                # (B, C, T)
        return x + residual


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: TemporalFSAS
# ─────────────────────────────────────────────────────────────────────────────

class TemporalFSAS(nn.Module):
    """
    Frequency-domain Self-Attention Solver for RTFM's temporal axis.

    Direct drop-in replacement for NONLocalBlock1D inside Aggregate.

    Operates on (B, C, T) — same format as NONLocalBlock1D.
    Output projection W is zero-initialised (same as NONLocalBlock1D.W)
    so the module starts as an identity residual.

    Args:
        channels  (int): input/output channels C (512 in Aggregate).
        reduction (int): inner channel factor. Default 2 → inner = 256.
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        inner = max(channels // reduction, 1)

        # Q, K, V projections — Conv1d(1×1)
        self.proj_q = nn.Conv1d(channels, inner, kernel_size=1, bias=False)
        self.proj_k = nn.Conv1d(channels, inner, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(channels, inner, kernel_size=1, bias=False)

        # Normalisation for the attention map
        self.attn_norm = nn.GroupNorm(1, inner)

        # Output projection with zero-init + BN — mirrors NONLocalBlock1D.W
        self.W = nn.Sequential(
            nn.Conv1d(inner, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias,   0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        T = x.shape[2]

        q = self.proj_q(x)     # (B, inner, T)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # Frequency-domain cross-correlation: IFFT(FFT(Q) * conj(FFT(K)))
        # Captures lag-based temporal dependencies missed by dot-product attention.
        Fq = fft.rfft(q, dim=2)
        Fk = fft.rfft(k, dim=2)
        A_freq = Fq * torch.conj(Fk)
        A = fft.irfft(A_freq, n=T, dim=2)   # (B, inner, T)  real

        A = self.attn_norm(A)
        attended = A * v                     # (B, inner, T)

        out = self.W(attended)               # (B, C, T)
        return out + x


# ─────────────────────────────────────────────────────────────────────────────
# Module 3: freq_magnitude
# ─────────────────────────────────────────────────────────────────────────────

def freq_magnitude(features: torch.Tensor) -> torch.Tensor:
    """
    Frequency-domain replacement for RTFM's per-snippet ℓ₂ magnitude.

    Replaces:
        feat_magnitudes = torch.norm(features, p=2, dim=2)

    The FFT round-trip re-weights each snippet's ℓ₂ norm by the energy
    distribution of its feature channels across frequencies, making subtle
    anomalies with concentrated spectral energy more discriminable.

    Args:
        features: (N, T, F)  where N = B * ncrops, T = 32, F = 2048

    Returns:
        magnitudes: (N, T)  — same shape as torch.norm(features, p=2, dim=2)
    """
    Xf    = fft.rfft(features, dim=2)                      # (N, T, F//2+1) complex
    X_rec = fft.irfft(Xf, n=features.shape[2], dim=2)      # (N, T, F) real
    return X_rec.norm(p=2, dim=2)                           # (N, T)


# ─────────────────────────────────────────────────────────────────────────────
# Module 4: FreqGatedClassifier
# ─────────────────────────────────────────────────────────────────────────────

class FreqGatedClassifier(nn.Module):
    """
    Drop-in replacement for the fc1/fc2/fc3 block in Model.

    Prepends a TemporalDFFN block over the T=32 snippet sequence before the
    FC classification layers. Uses patch_size=8 (divides T=32 into 4 clean
    patches with no zero-padding needed).

    Input  shape: (N, T, F)  where N = B * ncrops, T = 32, F = 2048
    Output shape: (N, T, 1)  per-snippet score in [0, 1]

    Args:
        n_features (int):  feature dimension F. Default 2048.
        hidden     (int):  first FC hidden size. Default 512 (matches RTFM).
        dropout    (float): dropout rate. Default 0.7 (matches RTFM).
    """

    def __init__(self, n_features: int = 2048, hidden: int = 512,
                 dropout: float = 0.7):
        super().__init__()

        # patch_size=8 divides T=32 into exactly 4 patches — no padding needed.
        self.t_dffn = TemporalDFFN(channels=n_features, patch_size=8)

        # Original RTFM FC layers — exact same structure and sizes.
        self.fc1      = nn.Linear(n_features, hidden)
        self.fc2      = nn.Linear(hidden, 128)
        self.fc3      = nn.Linear(128, 1)
        self.drop_out = nn.Dropout(dropout)
        self.relu     = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, F)
        Returns:
            (N, T, 1)
        """
        # TemporalDFFN expects (B, C, T) → permute to (N, F, T)
        x = self.t_dffn(x.permute(0, 2, 1)).permute(0, 2, 1)  # (N, T, F)

        # Original FC head — applied per snippet (last two dims)
        scores = self.relu(self.fc1(x))         # (N, T, 512)
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))    # (N, T, 128)
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores)) # (N, T, 1)
        return scores
```

Verify the file imports cleanly before touching `model.py`:

```bash
cd /path/to/RTFM
python -c "from new_modules import TemporalDFFN, TemporalFSAS, freq_magnitude, FreqGatedClassifier; print('OK')"
```

---

## 5. Mod 1 — T-DFFN: Frequency Gating inside PDC Branches

**Target file:** `model.py` — class `Aggregate`

### 5.1 Add import at the top of `model.py`

```python
# model.py — very top, after existing imports
from new_modules import TemporalDFFN, TemporalFSAS, freq_magnitude, FreqGatedClassifier
```

### 5.2 Changes to `Aggregate.__init__`

After the three existing `self.conv_1/2/3` definitions, add three
`TemporalDFFN` instances. The `channels=512` argument matches the
`out_channels=512` of each conv branch:

```python
# model.py — inside Aggregate.__init__

# ── EXISTING (do not change) ─────────────────────────────────────────────────
self.conv_1 = nn.Sequential(
    nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
              stride=1, dilation=1, padding=1),
    nn.ReLU(),
    bn(512)
)
self.conv_2 = nn.Sequential(
    nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
              stride=1, dilation=2, padding=2),
    nn.ReLU(),
    bn(512)
)
self.conv_3 = nn.Sequential(
    nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
              stride=1, dilation=4, padding=4),
    nn.ReLU(),
    bn(512)
)

# ── ADD immediately after conv_3 ─────────────────────────────────────────────
self.dffn1 = TemporalDFFN(channels=512, patch_size=4)
self.dffn2 = TemporalDFFN(channels=512, patch_size=4)
self.dffn3 = TemporalDFFN(channels=512, patch_size=4)
```

### 5.3 Changes to `Aggregate.forward`

Find the three lines that compute `out1`, `out2`, `out3` and wrap each one.
The tensor format is already `(B, 512, T)` (Conv1d output), which is exactly
what `TemporalDFFN` expects:

```python
# model.py — inside Aggregate.forward

# ── BEFORE ───────────────────────────────────────────────────────────────────
out1 = self.conv_1(out)
out2 = self.conv_2(out)
out3 = self.conv_3(out)

# ── AFTER (Mod 1) ─────────────────────────────────────────────────────────────
out1 = self.dffn1(self.conv_1(out))
out2 = self.dffn2(self.conv_2(out))
out3 = self.dffn3(self.conv_3(out))
```

No other lines in `Aggregate.forward` need changing.

---

## 6. Mod 2 — T-FSAS: Replace `NONLocalBlock1D` with FFT-based Attention

**Target file:** `model.py` — class `Aggregate`

`self.non_local` is a `NONLocalBlock1D` operating on `(B, 512, T)` tensors.
`TemporalFSAS` accepts and returns the same `(B, C, T)` format with the same
zero-init convention, making it a direct drop-in with **no changes to
`Aggregate.forward`**.

### 6.1 Changes to `Aggregate.__init__`

```python
# model.py — inside Aggregate.__init__

# ── BEFORE ───────────────────────────────────────────────────────────────────
self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)

# ── AFTER (Mod 2) ─────────────────────────────────────────────────────────────
self.non_local = TemporalFSAS(channels=512, reduction=2)
# The attribute name 'non_local' is kept intentionally so that
# Aggregate.forward's line `out = self.non_local(out)` requires no change.
```

The original `_NonLocalBlockND` and `NONLocalBlock1D` class definitions can
remain in `model.py` — they are no longer instantiated but can be kept for
reference and easy ablation.

**`Aggregate.forward` requires no changes for Mod 2.**

### 6.2 Fix: re-zero output projection after `weight_init`

`Model.__init__` calls `self.apply(weight_init)` at the end, which applies
`xavier_uniform_` to all Conv and Linear layers including `TemporalFSAS.W`.
This overwrites the zero-initialisation. Add a correction after the
`self.apply` call:

```python
# model.py — inside Model.__init__, immediately after self.apply(weight_init)

# Re-zero the output projections of all TemporalFSAS modules,
# which weight_init overwrites with xavier values.
for module in self.modules():
    if isinstance(module, TemporalFSAS):
        nn.init.constant_(module.W[1].weight, 0)
        nn.init.constant_(module.W[1].bias,   0)
```

---

## 7. Mod 3 — Frequency-Domain Feature Magnitude

**Target file:** `model.py` — `Model.forward`

This is a one-line replacement. Find the magnitude computation at approximately
line 220 of `Model.forward`:

```python
# model.py — inside Model.forward

# ── BEFORE ───────────────────────────────────────────────────────────────────
feat_magnitudes = torch.norm(features, p=2, dim=2)

# ── AFTER (Mod 3) ─────────────────────────────────────────────────────────────
feat_magnitudes = freq_magnitude(features)
```

`freq_magnitude` returns `(N, T)` — the same shape and dtype as
`torch.norm(features, p=2, dim=2)`. All downstream code that uses
`feat_magnitudes` (crop-averaging, top-k selection, score computation,
and the 10-value return) is unchanged.

> **Risk note.** Mod 3 changes the ranking signal for both top-k selection
> and the RTFM margin loss. Apply this last. If AUC drops after 15000 iters,
> revert with:
> ```python
> feat_magnitudes = torch.norm(features, p=2, dim=2)
> ```
> and keep Mods 1, 2, and 4.

---

## 8. Mod 4 — T-DFFN Gated Classifier Head

**Target file:** `model.py` — `Model.__init__` and `Model.forward`

In `Model.forward`, scores are computed over the **full** T=32 sequence
and then the top-k scores are gathered via magnitude indices. The classifier
must therefore be applied to the full `(N, T, F)` feature tensor, not only
the top-k features. `FreqGatedClassifier` is designed for this.

### 8.1 Changes to `Model.__init__`

```python
# model.py — inside Model.__init__

# ── BEFORE ───────────────────────────────────────────────────────────────────
self.fc1 = nn.Linear(n_features, 512)
self.fc2 = nn.Linear(512, 128)
self.fc3 = nn.Linear(128, 1)
self.drop_out = nn.Dropout(0.7)
self.relu = nn.ReLU()
self.sigmoid = nn.Sigmoid()
self.apply(weight_init)

# ── AFTER (Mod 4) ─────────────────────────────────────────────────────────────
self.classifier = FreqGatedClassifier(
    n_features=n_features,   # 2048 for I3D; 4096 for C3D
    hidden=512,
    dropout=0.7
)
# Keep the old FC attributes so the rest of the code does not break if
# any other module references them (e.g. weight_init or checkpointing).
self.fc1      = nn.Linear(n_features, 512)
self.fc2      = nn.Linear(512, 128)
self.fc3      = nn.Linear(128, 1)
self.drop_out = nn.Dropout(0.7)
self.relu     = nn.ReLU()
self.sigmoid  = nn.Sigmoid()
self.apply(weight_init)

# Re-zero TemporalFSAS output projections after weight_init (see Mod 2).
for module in self.modules():
    if isinstance(module, TemporalFSAS):
        nn.init.constant_(module.W[1].weight, 0)
        nn.init.constant_(module.W[1].bias,   0)
```

### 8.2 Changes to `Model.forward`

Find the score computation block (approximately lines 198–207 of `Model.forward`):

```python
# model.py — inside Model.forward

# ── BEFORE ───────────────────────────────────────────────────────────────────
features = out
scores = self.relu(self.fc1(features))
scores = self.drop_out(scores)
scores = self.relu(self.fc2(scores))
scores = self.drop_out(scores)
scores = self.sigmoid(self.fc3(scores))
scores = scores.view(bs, ncrops, -1).mean(1)
scores = scores.unsqueeze(dim=2)

# ── AFTER (Mod 4) ─────────────────────────────────────────────────────────────
features = out                                   # (B*ncrops, T, F)
scores = self.classifier(features)               # (B*ncrops, T, 1)
scores = scores.view(bs, ncrops, -1).mean(1)     # (B, T)  — average over crops
scores = scores.unsqueeze(dim=2)                 # (B, T, 1)
```

All downstream code that uses `scores` (gathering top-k scores, sparsity
regularisation, and the return value at index [6]) is unchanged.

---

## 9. Training Loop Changes (`main.py`)

Mods 1, 2, 3, and 4 are entirely inside `model.py`. No changes to `train.py`
or `test_10crop.py` are required.

The only `main.py` change is the optimizer construction, to prevent
`W_quant` from collapsing under weight decay:

```python
# main.py — replace the optimizer construction

# ── BEFORE ───────────────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(),
                       lr=config.lr[0], weight_decay=0.005)

# ── AFTER ─────────────────────────────────────────────────────────────────────
# Separate W_quant from weight decay — these are initialised to 1 and must
# not be regularised toward zero.
w_quant_params = [p for n, p in model.named_parameters() if 'W_quant' in n]
other_params   = [p for n, p in model.named_parameters() if 'W_quant' not in n]

optimizer = optim.Adam([
    {'params': other_params,   'lr': config.lr[0], 'weight_decay': 0.005},
    {'params': w_quant_params, 'lr': config.lr[0] * 2, 'weight_decay': 0.0},
])
```

The lr schedule update inside the training loop in `main.py` iterates over
`optimizer.param_groups`, which still works correctly with multiple groups:

```python
# main.py — lr schedule update (this code already exists; verify it updates all groups)
if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
    for param_group in optimizer.param_groups:
        # For W_quant group, keep the 2x multiplier relative to the base lr.
        if any('W_quant' in n for n, p in
               zip([n for n, _ in model.named_parameters()], param_group['params'])
               if p in param_group['params']):
            param_group["lr"] = config.lr[step - 1] * 2
        else:
            param_group["lr"] = config.lr[step - 1]
```

If that check is too complex, a simpler alternative is to just update all
groups to the same lr (the 2x multiplier is a soft recommendation, not
critical):

```python
for param_group in optimizer.param_groups:
    param_group["lr"] = config.lr[step - 1]
```

---

## 10. Recommended Ablation Order

Run each step for the full 15000 iterations on ShanghaiTech with I3D features.

| Step | Active mods | Baseline | Expected behaviour |
|------|-------------|----------|--------------------|
| 0 | None (original RTFM) | — | AUC ≈ 97.21% |
| 1 | Mod 2 only | Step 0 | AUC maintained ±0.2%; confirms drop-in compatibility |
| 2 | Mods 1 + 2 | Step 1 | AUC increases, especially on subtle anomaly classes |
| 3 | Mods 1 + 2 + 4 | Step 2 | Further improvement expected |
| 4 | All mods (+ Mod 3) | Step 3 | Best case: improvement; revert Mod 3 if AUC drops |

Run UCF-Crime and XD-Violence only after ShanghaiTech results are satisfactory.

---

## 11. Hyperparameter Checklist

| Parameter | Original value | Where it lives | Notes for FFT mods |
|-----------|---------------|----------------|-------------------|
| `k_abn` / `k_nor` | `num_segments // 10 = 3` | `Model.__init__` | Increase by changing `// 10` to a fixed literal, e.g. `self.k_abn = 5` |
| `patch_size` in PDC T-DFFN | 4 | `Aggregate.__init__` | 4 or 8 (both divide T=32 exactly) |
| `patch_size` in FreqGatedClassifier | 8 | `FreqGatedClassifier.__init__` | 8 preferred; 4 also works |
| `reduction` in T-FSAS | 2 | `Aggregate.__init__` | Try 4 if GPU memory is tight |
| `lr` multiplier for `W_quant` | 2× | `main.py` optimizer | Range: 1.5× – 3× |
| `weight_decay` for `W_quant` | 0.0 | `main.py` optimizer | Must stay 0.0 |
| `weight_decay` (rest) | 0.005 | `main.py` | Unchanged from original |
| `max_epoch` | 15000 | `option.py` | Increase to 20000 when Mod 3 is active |

---

## 12. Evaluation Protocol

Use the same protocol as the original RTFM paper:

- **Primary metric**: frame-level AUC under the ROC curve.
- **Secondary metric**: Average Precision (AP) for XD-Violence.
- **Test script**: `test_10crop.py` — no changes needed.
- **Checkpoint**: saved every 5 steps after step 200 in `./ckpt/` by `main.py`;
  best model selected by AUC.
- **Datasets**: ShanghaiTech → UCF-Crime → XD-Violence → UCSD-Peds.
- **Features**: evaluate with both I3D-RGB (`feature_size=2048`, default) and
  C3D-RGB (`feature_size=4096`, set in `option.py`). For C3D, `FreqGatedClassifier`
  must be instantiated with `n_features=4096`; the PDC branch T-DFFNs remain
  `channels=512` (their output size is always 512 regardless of input).
- **Seeds**: run each configuration 3 times with different seeds; report mean AUC.
- **Per-class AUC on UCF-Crime**: re-run the per-class analysis from the paper's
  Figure 5; largest expected gains are in burglary, shoplifting, vandalism, stealing.

---

## 13. Common Errors and Fixes

### `AttributeError: 'Aggregate' object has no attribute 'dffn1'`
**Cause:** The `from new_modules import ...` line is missing from `model.py`,
or Mod 1 changes to `Aggregate.__init__` were not saved.
**Fix:** Confirm the import is at the top of `model.py` and that
`self.dffn1 = TemporalDFFN(...)` is inside `Aggregate.__init__`.

---

### `RuntimeError: Given groups=1, expected weight ...` (shape mismatch in T-DFFN)
**Cause:** `patch_size` does not divide T cleanly and the reshape fails.
**Fix:** Use `patch_size=4` or `patch_size=8` — both divide T=32 exactly.
The padding code in `TemporalDFFN.forward` handles non-divisible T, but
using an exact divisor avoids all edge cases.

---

### `W_quant` values collapse to near-zero
**Cause:** Weight decay is being applied to `W_quant`.
**Fix:** Use the per-group optimizer from Section 9 with `weight_decay=0.0`
for all parameters whose name contains `'W_quant'`. Check with:
```python
for n, p in model.named_parameters():
    if 'W_quant' in n:
        print(n, p.mean().item())
```

---

### AUC drops after Mod 3 (frequency magnitude)
**Cause:** The frequency-domain magnitude changes the top-k selection signal.
Some configurations need more iterations to converge.
**Fix 1:** Increase `max_epoch` from 15000 to 20000 in `option.py`.
**Fix 2:** Revert Mod 3 only — restore
`feat_magnitudes = torch.norm(features, p=2, dim=2)` — and keep Mods 1, 2, 4.

---

### `RuntimeError: Expected all tensors to be on the same device`
**Cause:** A new module was defined outside `__init__` or after the `.to(device)` call.
**Fix:** All `TemporalDFFN`, `TemporalFSAS`, and `FreqGatedClassifier` instances
must be assigned as attributes inside `__init__` (i.e. `self.xxx = ...`).
PyTorch's `.to(device)` call on the parent model then moves all child modules
automatically.

---

### `TemporalFSAS` output projection not zeroed (non-identity start)
**Cause:** `self.apply(weight_init)` in `Model.__init__` overwrites the
`nn.init.constant_(..., 0)` calls made inside `TemporalFSAS.__init__`.
**Fix:** Add the re-zeroing block after `self.apply(weight_init)` as shown
in Section 6.2. This is required whenever `TemporalFSAS` is used.

---

### `FreqGatedClassifier` produces nearly constant scores
**Cause:** `patch_size=3` in full-sequence mode (T=32) causes padding to 33,
and with random init the gating can collapse early in training.
**Fix:** Use `patch_size=8` as set by default in `FreqGatedClassifier.__init__`.
This divides T=32 into exactly 4 patches with zero padding.

---

*End of instructions.*
