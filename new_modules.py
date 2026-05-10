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
    Complex-valued temporal spectral filter for RTFM's PDC branch outputs.

    v4 — literature-grounded redesign:

    FEDformer (ICML 2022) showed that frequency-domain weights should be
    complex-valued (torch.cfloat), capturing both amplitude AND phase.
    Phase shifts let the filter model temporal offsets between channels —
    impossible with real-only scaling.  FEDformer also proved that diverse
    mode coverage (all bins, not just low-pass) outperforms low-pass only.

    AFNO (ICLR 2022) showed that applying the filter to the COMBINED
    multi-scale output (all dilation branches together) captures cross-branch
    frequency correlations that per-branch filters miss.  This module is
    therefore instantiated ONCE on the 1536-channel concatenated PDC output
    in Aggregate, not three times on individual 512-channel branches.

    Parameterisation — complex delta filter:
        W = 1 + ΔW,  where ΔW = W_quant[...,0] + j·W_quant[...,1]
    Init: W_quant=0 → ΔW=0+0j → W=1+0j → EXACT identity, gradient flows
    from step 0.  Weight decay (wd=1e-3 in main.py) pulls ΔW back toward 0
    (identity), regularising the filter against spectral overfitting.

    Args:
        channels  (int): input channels — 1536 for concatenated PDC output.
        patch_size(int): kept for API compatibility; unused.
        n_bins    (int): rfft bins to filter (default 17 = all bins for T=32).
                         Bins beyond n_bins pass through unchanged (scale=1).
    """

    def __init__(self, channels: int, patch_size: int = 4, n_bins: int = 17):
        super().__init__()
        self.n_bins = n_bins

        # Complex delta: W_quant[..., 0]=real_delta, W_quant[..., 1]=imag_delta.
        # All zeros init → ΔW=0+0j → W=1+0j → identity.
        # Named W_quant so main.py optimizer treatment (lr×2, wd=1e-3) applies.
        self.W_quant = nn.Parameter(torch.zeros(1, channels, n_bins, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) — exact identity when W_quant=0
        """
        B, C, T = x.shape
        actual_bins = T // 2 + 1
        n = min(self.n_bins, actual_bins)   # bins covered by the learned filter

        xf = fft.rfft(x, dim=2)            # (B, C, actual_bins) complex

        # Complex filter W = 1 + ΔW (identity at init, regularised by wd)
        W_delta = torch.view_as_complex(
            self.W_quant[:, :, :n, :].contiguous()
        )                                   # (1, C, n) complex
        W = 1.0 + W_delta                   # (1, C, n) complex — W=1+0j at init

        # Apply filter to the first n bins; high bins pass through unchanged.
        # torch.cat with an empty tensor is a no-op when n == actual_bins.
        xf_out = torch.cat([xf[:, :, :n] * W, xf[:, :, n:]], dim=2)

        return fft.irfft(xf_out, n=T, dim=2)   # (B, C, T)


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
        inner = self.proj_q.out_channels
        scale = inner ** -0.5   # temperature: same role as /sqrt(d_k) in attention

        q = self.proj_q(x) * scale  # (B, inner, T)
        k = self.proj_k(x)
        v = self.proj_v(x)

        Fq = fft.rfft(q, dim=2)     # (B, inner, n_bins) complex
        Fk = fft.rfft(k, dim=2)
        Fv = fft.rfft(v, dim=2)

        # Correct FSAS (FFTFormer, CVPR 2023):
        #   Attended = IFFT( FFT(Q) ⊙ conj(FFT(K)) ⊙ FFT(V) )
        # All three stay in frequency domain — equivalent to circular
        # convolution-attention: (Q⋆K) * V  (cross-corr weighted sum of V).
        # The original bug converted A to time domain before multiplying V,
        # giving A[t]*V[t] (elementwise scale) instead of Σ_τ A[τ]·V[t-τ].
        attended_freq = Fq * torch.conj(Fk) * Fv   # (B, inner, n_bins) complex
        attended = fft.irfft(attended_freq, n=T, dim=2)  # (B, inner, T) real

        attended = self.attn_norm(attended)

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
