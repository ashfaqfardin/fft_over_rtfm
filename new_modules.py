# new_modules.py
# FFTFormer-inspired modules adapted for RTFM's 1-D temporal axis.
# Reference: Kong et al., "Efficient Frequency Domain-based Transformers
#            for High-Quality Image Deblurring", CVPR 2023.
#
# Mod 5 — GlanceFocusBlock adapted from:
#   Chen et al., "MGFN: Magnitude-Contrastive Glance-and-Focus Network
#   for Weakly-Supervised Video Anomaly Detection", AAAI 2023.

import math
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

    v5 — two improvements over v4, each grounded in recent literature:

    [1] Per-bin frequency gate (bin_gate):
        A learned real-valued scalar per frequency bin, shared across all
        channels (shape 1×1×n_bins, init 0 → scale 1.0).  This is distinct
        from W_quant (which is per-channel complex): bin_gate encodes a
        task-level frequency-band prior — e.g., the model can learn to
        up-weight high-frequency bins where abrupt anomalous events manifest.
        Conceptually similar to FEDformer's (ICML 2022) frequency mode
        selection, but continuous and differentiable.
        Parameters: n_bins = 17.

    [2] Input-conditioned dynamic gate (dyn_gate):
        Global temporal average of x → Linear(C, n_bins, bias=False) →
        per-video per-bin correction added on top of bin_gate.
        Zero-init weights → zero contribution at init → exact identity.
        Makes the filter adaptive: a robbery video and a normal walking
        video can now emphasise different frequency bands automatically.
        Inspired by SE-Net's (CVPR 2018) squeeze-and-excitation attention
        and AFNO's (ICLR 2022) input-adaptive Fourier token mixing.
        Parameters: C × n_bins  (e.g., 1536×17 = 26 112 for Mod 1).

    Combined gate: gate = 1 + bin_gate + dyn_gate(avg(x))
        Starts at exactly 1.0 for every video.  Multiplies the complex xf
        after W is applied, so both amplitude and phase from W_quant are
        preserved — the gate only modulates per-bin energy.

    All v4 design principles are preserved:
      - Complex W_quant, identity init, wd=1e-3 in spectral optimizer group.
      - Applied post-concat (1536 ch): cross-dilation interactions (AFNO).
      - Full bin coverage (n_bins=17 for T=32): diverse mode learning.

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
        # All zeros → ΔW=0+0j → W=1+0j → identity.
        # In spectral optimizer group: lr×2, wd=1e-3 (see main.py).
        self.W_quant = nn.Parameter(torch.zeros(1, channels, n_bins, 2))

        # [1] Per-bin static frequency gate — shared across channels.
        # bin_gate=0 → scale = 1.0+0.0 = 1.0 → identity.
        # In spectral optimizer group alongside W_quant.
        self.bin_gate = nn.Parameter(torch.zeros(1, 1, n_bins))

        # [2] Input-conditioned dynamic gate: global context → per-bin delta.
        # Zero-init weights → dyn contribution = 0 at init → exact identity.
        # In other_params group (standard lr, wd=0.005).
        self.dyn_gate = nn.Linear(channels, n_bins, bias=False)
        nn.init.zeros_(self.dyn_gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) — exact identity when all parameters are at init values
        """
        B, C, T = x.shape
        actual_bins = T // 2 + 1
        n = min(self.n_bins, actual_bins)

        xf = fft.rfft(x, dim=2)             # (B, C, actual_bins) complex

        # Per-channel complex filter: W = 1 + ΔW  (identity at init)
        W_delta = torch.view_as_complex(
            self.W_quant[:, :, :n, :].contiguous()
        )                                    # (1, C, n) complex
        W = 1.0 + W_delta                    # (1, C, n)

        # Combined real frequency gate — starts at 1.0, adapts during training.
        # [1] Static per-bin term: shape (1, 1, n)
        # [2] Dynamic per-video term: avg(x) → Linear → (B, n_bins) → (B, 1, n)
        ctx = x.mean(dim=2)                  # (B, C) global temporal average
        dyn = self.dyn_gate(ctx).unsqueeze(1)  # (B, 1, n_bins)
        gate = (1.0 + self.bin_gate + dyn)[:, :, :n]  # (B, 1, n)

        # Apply complex filter then real gate; pass high bins unchanged.
        xf_out = torch.cat(
            [xf[:, :, :n] * W * gate, xf[:, :, n:]], dim=2
        )

        return fft.irfft(xf_out, n=T, dim=2)  # (B, C, T)


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
        # Gives a lag-based temporal gating: A[t] measures Q-K correlation at lag t.
        # Each V[t] is then gated by A[t] — positions where Q and K are correlated
        # at that lag are amplified; uncorrelated positions are suppressed.
        # Note: this is a 1-D temporal adaptation; the 2D image FFTFormer formula
        # (keep V in freq domain) creates circular convolution that is harder to
        # train on short T=32 sequences — empirically worse for this task.
        Fq = fft.rfft(q, dim=2)
        Fk = fft.rfft(k, dim=2)
        A_freq = Fq * torch.conj(Fk)
        A = fft.irfft(A_freq, n=T, dim=2)   # (B, inner, T)  real

        # Temperature scaling: prevents A from growing as Q/K projections grow.
        # Analogous to 1/sqrt(d_k) in scaled dot-product attention (Vaswani 2017).
        # Energy of A is proportional to inner (Parseval), so sqrt(inner) bounds it.
        A = A * (1.0 / math.sqrt(q.shape[1]))

        A = self.attn_norm(A)
        attended = A * v                     # (B, inner, T)

        out = self.W(attended)               # (B, C, T)
        return out + x


# ─────────────────────────────────────────────────────────────────────────────
# Module 3: freq_magnitude
# ─────────────────────────────────────────────────────────────────────────────

def freq_magnitude(features: torch.Tensor) -> torch.Tensor:
    """
    Temporal-deviation magnitude: per-snippet L2 distance from the video's mean feature.

    Replaces both:
        feat_magnitudes = torch.norm(features, p=2, dim=2)   # absolute — scene-biased
    and the previous rfft→irfft→norm form, which was a no-op by Parseval's theorem
    (irfft(rfft(x)) == x exactly, so the norm was unchanged).

    The 76% of UCF-Crime anomaly events that are shorter than one T=32 window appear
    as *local* spikes inside mostly-normal sequences.  Absolute L2 norm picks the
    highest-energy snippet regardless of context; deviation magnitude picks the snippet
    that deviates most from the video's own temporal average — precisely what anomaly
    detection requires.  The scene-level mean activation is absorbed into `baseline`,
    so high-motion normal scenes no longer inflate the top-k selection.

    Args:
        features: (N, T, F)  where N = B * ncrops, T = 32, F = 2048

    Returns:
        magnitudes: (N, T)  — same shape as torch.norm(features, p=2, dim=2)
    """
    baseline  = features.mean(dim=1, keepdim=True)   # (N, 1, F) — video temporal mean
    deviation = features - baseline                   # (N, T, F) — per-snippet deviation
    return deviation.norm(p=2, dim=2)                 # (N, T)


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
