# new_modules.py
# FFTFormer-inspired modules adapted for RTFM's 1-D temporal axis.
# Reference: Kong et al., "Efficient Frequency Domain-based Transformers
#            for High-Quality Image Deblurring", CVPR 2023.

import torch
import torch.nn as nn
import torch.fft as fft


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 & 4 helper: TemporalDFFN
# ─────────────────────────────────────────────────────────────────────────────

class TemporalDFFN(nn.Module):
    """
    Discriminative Frequency-domain FFN adapted for RTFM's temporal axis.

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
