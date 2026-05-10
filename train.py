import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def smooth(arr, lamda1, gate_thresh=0.3):
    # Boundary-aware temporal smoothness (Improvement 5).
    # Transitions that already differ by > gate_thresh are down-weighted so
    # genuine anomaly boundaries are not over-penalised.
    diff = arr[1:] - arr[:-1]                              # (T-1,)
    gate = torch.sigmoid(gate_thresh - torch.abs(diff).detach())
    return lamda1 * torch.sum(gate * diff ** 2)


def smooth_per_video(arr_2d, lamda1, gate_thresh=0.3):
    # Correct per-video smoothness: arr_2d is (B, T).
    # Each row is one video — no cross-video boundary penalties.
    diff = arr_2d[:, 1:] - arr_2d[:, :-1]                 # (B, T-1)
    gate = torch.sigmoid(gate_thresh - torch.abs(diff).detach())
    return lamda1 * torch.sum(gate * diff ** 2)


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__l1_loss__ = nn.MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(-torch.abs(x))
        return torch.abs(torch.mean(-x * target + torch.clamp(x, min=0) + torch.log(tmp)))


# ── Loss 1 (default): RTFM ───────────────────────────────────────────────────
class RTFM_loss(nn.Module):
    """Original RTFM loss: BCE on bag scores + feature-magnitude ranking term."""
    def __init__(self, alpha=0.0001, margin=100):
        super().__init__()
        self.alpha  = alpha
        self.margin = margin
        self.criterion = nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        label = label.to(score.device)
        loss_cls = self.criterion(score, label)
        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
        return loss_cls + self.alpha * loss_rtfm


# ── Loss 2: Ranking (Sultani MIL hinge) ──────────────────────────────────────
class RankingLoss(nn.Module):
    """MIL hinge ranking loss from Sultani et al. (2018).
    Top-k abnormal segments must outscore top-k normal segments by at least `margin`.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        return F.relu(self.margin - score_abnormal + score_normal).mean()


# ── Loss 3: Focal BCE ─────────────────────────────────────────────────────────
class FocalBCELoss(nn.Module):
    """Focal loss (Lin et al. 2017) replacing plain BCE to handle class imbalance.
    gamma=2 down-weights easy normal segments so training focuses on hard cases.
    Retains the RTFM feature-magnitude term for stable convergence.
    """
    def __init__(self, alpha=0.0001, margin=100, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha_rtfm = alpha
        self.margin     = margin
        self.gamma      = gamma
        self.eps        = eps

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        label = label.to(score.device)

        pt        = torch.where(label == 1, score, 1 - score).clamp(self.eps, 1 - self.eps)
        loss_cls  = -(((1 - pt) ** self.gamma) * torch.log(pt)).mean()

        loss_abn  = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
        loss_nor  = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
        return loss_cls + self.alpha_rtfm * loss_rtfm


# ── Loss 4: Contrastive ───────────────────────────────────────────────────────
class ContrastiveLoss(nn.Module):
    """Cosine-margin contrastive loss.
    Forces cosine similarity between L2-normalised normal and abnormal feature
    centroids to stay below (1 - margin), pushing the two distributions apart.
    """
    def __init__(self, alpha=1.0, margin=0.5):
        super().__init__()
        self.alpha    = alpha
        self.margin   = margin
        self.criterion = nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        label = label.to(score.device)
        loss_cls = self.criterion(score, label)

        n_mean       = F.normalize(feat_n.mean(dim=1), dim=1)   # (B, F)
        a_mean       = F.normalize(feat_a.mean(dim=1), dim=1)   # (B, F)
        cos_sim      = (n_mean * a_mean).sum(dim=1)              # (B,)
        loss_contrast = F.relu(cos_sim - (1.0 - self.margin)).mean()
        return loss_cls + self.alpha * loss_contrast


# ── Loss 5: MGFN Magnitude Contrastive ───────────────────────────────────────
class MGFNMagnitudeLoss(nn.Module):
    """Scene-adaptive magnitude contrastive loss from MGFN (AAAI 2023).

    Extends the RTFM magnitude term with two batch-level signals:
      - Inter-class separation: abnormal mags must exceed normal mags by margin/2.
      - Intra-class compactness: variance within each class is penalised so the
        magnitude distribution is tight and scene-independent.
    """
    def __init__(self, alpha=0.0001, margin=100, mc_weight=0.1):
        super().__init__()
        self.alpha     = alpha
        self.margin    = margin
        self.mc_weight = mc_weight
        self.criterion = nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        loss_cls = self.criterion(score, label.to(score.device))

        # Original RTFM magnitude ranking term
        loss_abn  = torch.abs(self.margin - torch.norm(feat_a.mean(1), p=2, dim=1))
        loss_nor  = torch.norm(feat_n.mean(1), p=2, dim=1)
        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        # MGFN batch-level magnitude contrastive
        mag_a = torch.norm(feat_a.mean(1), p=2, dim=1)   # (B,)
        mag_n = torch.norm(feat_n.mean(1), p=2, dim=1)   # (B,)
        inter      = F.relu(self.margin / 2 - (mag_a - mag_n)).mean()
        intra_comp = torch.var(mag_a) + torch.var(mag_n)
        loss_mc    = inter + 0.5 * intra_comp

        return loss_cls + self.alpha * loss_rtfm + self.mc_weight * loss_mc


# ── Loss 6: Deviation MIL — co-designed for --mod 3 ─────────────────────────
class DeviationMIL_loss(nn.Module):
    """Co-designed loss for --mod 3 deviation features.

    Model.forward passes deviation features (feat - temporal_mean) as feat_n/feat_a
    when Mod 3 is active.  This loss ranks them with a scale-invariant hinge:

        relu(margin - (||dev_abn|| - ||dev_nor||))

    This directly optimises the gap between the deviation magnitudes of selected
    abnormal and normal snippets, without assuming any absolute scale.

    Do NOT use this loss without --mod 3: when model.py passes raw features,
    the hinge margin of 1.0 may be too small relative to absolute L2 norms (~7-10).
    Do NOT use --loss rtfm with --mod 3: that loss measures absolute L2 norm but
    selection uses deviation magnitude — the two are misaligned.

    Args:
        alpha  (float): weight of the deviation hinge term.  Default 0.1.
        margin (float): minimum required gap between abnormal and normal deviation
                        magnitudes.  Default 1.0 — chosen relative to typical
                        deviation norms (~0.5–3.0 for I3D post-Aggregate features).
    """
    def __init__(self, alpha: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.alpha     = alpha
        self.margin    = margin
        self.criterion = nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        loss_cls = self.criterion(score, label.to(score.device))

        # Deviation magnitude of the selected snippet centroids
        mag_abn = feat_a.mean(dim=1).norm(p=2, dim=1)   # (B,)
        mag_nor = feat_n.mean(dim=1).norm(p=2, dim=1)   # (B,)

        # Hinge: abnormal deviation must exceed normal deviation by at least margin.
        # Scale-invariant — works regardless of absolute feature magnitude.
        loss_dev = F.relu(self.margin - (mag_abn - mag_nor)).mean()

        return loss_cls + self.alpha * loss_dev


_LOSS_REGISTRY = {
    'rtfm':        lambda: RTFM_loss(alpha=0.0001, margin=100),
    'ranking':     lambda: RankingLoss(margin=1.0),
    'focal':       lambda: FocalBCELoss(alpha=0.0001, margin=100, gamma=2.0),
    'contrastive': lambda: ContrastiveLoss(alpha=1.0, margin=0.5),
    'mgfn':        lambda: MGFNMagnitudeLoss(alpha=0.0001, margin=100, mc_weight=0.1),
    'deviation':   lambda: DeviationMIL_loss(alpha=0.1, margin=1.0),
}


def train(nloader, aloader, model, batch_size, optimizer, viz, device,
          loss_fn, smooth_weight=8e-4, sparse_weight=8e-3,
          pseudo_weight=0.0, pseudo_threshold=0.8,
          grad_clip=10.0):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)

        # scores_per_video: (B_nor + B_abn, T, 1) — kept before flattening
        scores_per_video = scores

        scores     = scores.view(batch_size * 32 * 2, -1).squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        # Bug fix: compute smoothness per-video (not across concatenated videos).
        # scores_per_video[batch_size:] are the abnormal videos, shape (B_abn, T, 1).
        abn_scores_2d = scores_per_video[batch_size:, :, 0]   # (B_abn, T)
        loss_sparse   = sparsity(abn_scores, batch_size, sparse_weight)
        loss_smooth   = smooth_per_video(abn_scores_2d, smooth_weight)

        cost = (loss_fn(score_normal, score_abnormal, nlabel, alabel,
                        feat_select_normal, feat_select_abn)
                + loss_smooth + loss_sparse)

        # Pseudo-label self-training (MIST, CVPR 2021) — active when pseudo_weight > 0
        # After warmup, high-confidence snippet predictions are used as extra supervision.
        if pseudo_weight > 0.0:
            nor_s = scores_per_video[:batch_size, :, 0]    # (B_nor, T)
            with torch.no_grad():
                abn_mask = abn_scores_2d > pseudo_threshold
                nor_mask = nor_s < (1.0 - pseudo_threshold)
            loss_pseudo = torch.tensor(0.0, device=device)
            if abn_mask.any():
                loss_pseudo = loss_pseudo + F.binary_cross_entropy(
                    abn_scores_2d[abn_mask], torch.ones(abn_mask.sum(), device=device)
                )
            if nor_mask.any():
                loss_pseudo = loss_pseudo + F.binary_cross_entropy(
                    nor_s[nor_mask], torch.zeros(nor_mask.sum(), device=device)
                )
            cost = cost + pseudo_weight * loss_pseudo

        viz.plot_lines('loss', cost.item())
        viz.plot_lines('smooth loss', loss_smooth.item())
        viz.plot_lines('sparsity loss', loss_sparse.item())
        optimizer.zero_grad()
        cost.backward()
        # Gradient clipping prevents extreme spikes (0 = disabled).
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
