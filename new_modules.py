# new_modules.py
# FFTFormer-inspired modules adapted for RTFM's 1-D temporal axis.
# Reference: Kong et al., "Efficient Frequency Domain-based Transformers
#            for High-Quality Image Deblurring", CVPR 2023.
#
# Mod 5 — GlanceFocusBlock adapted from:
#   Chen et al., "MGFN: Magnitude-Contrastive Glance-and-Focus Network
#   for Weakly-Supervised Video Anomaly Detection", AAAI 2023.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 & 4 helper: TemporalDFFN
# ─────────────────────────────────────────────────────────────────────────────

class TemporalDFFN(nn.Module):
    """
    Discriminative Frequency-domain FFN adapted for RTFM's temporal axis.

    v3: Log-domain spectral amplitude filter over the full T=32 sequence.

    scale = exp(W_quant).  W_quant=0 at init → scale=1 everywhere → the
    module is an EXACT identity from step 0.  Unlike the previous zero-init
    out_proj approach, the gradient reaches W_quant immediately from step 0
    (no zero-weight path blocking it).

    exp() keeps scale strictly positive (no phase inversions).
    Clamped to [-2, 2] so scale stays in [0.14, 7.39] — prevents runaway
    amplification while still allowing meaningful frequency selection.

    Low bins (0–4): learn to weight slow background patterns.
    High bins (8–16): learn to weight rapid anomaly bursts.
    Both complement the dilation convolutions which handle local spatial scale.

    Args:
        channels   (int): feature channels C (512 for PDC branches).
        patch_size (int): kept for API compatibility; unused.
    """

    def __init__(self, channels: int, patch_size: int = 4):
        super().__init__()
        n_bins = 17  # T=32 → rfft → T//2+1 = 17 bins

        # Log-spectral filter: W_quant=0 → exp(0)=1 → identity at init.
        # Named W_quant so main.py zero weight-decay / lr×2 treatment applies.
        self.W_quant = nn.Parameter(torch.zeros(1, channels, n_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)  — exact identity when W_quant=0
        """
        B, C, T = x.shape
        n_bins = T // 2 + 1

        xf = fft.rfft(x, dim=2)   # (B, C, n_bins) complex

        # exp(W_quant) for trained bins; exp(0)=1 for any extra high-freq bins
        # that appear in variable-length test videos (T > 32).
        w = self.W_quant
        w_bins = w.shape[2]
        if n_bins <= w_bins:
            scale = torch.exp(w[:, :, :n_bins].clamp(-2, 2))
        else:
            pad = torch.zeros(1, C, n_bins - w_bins, device=x.device, dtype=x.dtype)
            scale = torch.exp(torch.cat([w, pad], dim=2).clamp(-2, 2))

        xf = xf * scale
        return fft.irfft(xf, n=T, dim=2)   # (B, C, T)


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

        # Output projection with zero-init BN — mirrors NONLocalBlock1D.W
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
    Frequency-domain replacement for RTFM's per-snippet L2 magnitude.

    Replaces:
        feat_magnitudes = torch.norm(features, p=2, dim=2)

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


# ─────────────────────────────────────────────────────────────────────────────
# Module 5: GlanceFocusBlock  (MGFN, AAAI 2023)
# ─────────────────────────────────────────────────────────────────────────────

class GlanceFocusBlock(nn.Module):
    """
    Glance-and-Focus block adapted from MGFN (AAAI 2023) for 1-D temporal features.

    Glance branch: global channel attention via temporal average-pooling + MLP,
    producing a per-channel weight in (0,1) that re-scales all time-steps.

    Focus branch: local gated depthwise conv that sharpens short-range anomaly
    details without mixing channels (depthwise keeps parameter cost low).

    Both branches are summed and normalised, then added back to the residual
    so the block is near-identity at initialisation.

    Args:
        channels (int): number of input/output channels C (512 inside Aggregate).
    """

    def __init__(self, channels: int):
        super().__init__()

        # Glance: global temporal pooling → channel MLP → sigmoid gate
        self.glance_pool = nn.AdaptiveAvgPool1d(1)
        self.glance_proj = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

        # Focus: depthwise conv (one filter per channel) + pointwise gate
        self.focus_dw   = nn.Conv1d(channels, channels, kernel_size=3,
                                    padding=1, groups=channels, bias=False)
        self.focus_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        # Glance: channel attention from global pool
        g = self.glance_pool(x).squeeze(-1)        # (B, C)
        g = self.glance_proj(g).unsqueeze(-1)      # (B, C, 1)
        x_glance = x * g                           # (B, C, T)

        # Focus: gated depthwise local conv
        x_focus = torch.sigmoid(self.focus_gate(x)) * self.focus_dw(x)  # (B, C, T)

        return self.norm(x_glance + x_focus) + x   # residual keeps identity at init
