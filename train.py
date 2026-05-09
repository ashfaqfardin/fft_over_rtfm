import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    return lamda1 * torch.sum((arr2 - arr) ** 2)


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


_LOSS_REGISTRY = {
    'rtfm':        lambda: RTFM_loss(alpha=0.0001, margin=100),
    'ranking':     lambda: RankingLoss(margin=1.0),
    'focal':       lambda: FocalBCELoss(alpha=0.0001, margin=100, gamma=2.0),
    'contrastive': lambda: ContrastiveLoss(alpha=1.0, margin=0.5),
}


def train(nloader, aloader, model, batch_size, optimizer, viz, device,
          loss_name='rtfm', smooth_weight=8e-4, sparse_weight=8e-3):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)

        scores    = scores.view(batch_size * 32 * 2, -1).squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_fn     = _LOSS_REGISTRY[loss_name]().to(device)
        loss_sparse = sparsity(abn_scores, batch_size, sparse_weight)
        loss_smooth = smooth(abn_scores, smooth_weight)
        cost = (loss_fn(score_normal, score_abnormal, nlabel, alabel,
                        feat_select_normal, feat_select_abn)
                + loss_smooth + loss_sparse)

        viz.plot_lines('loss', cost.item())
        viz.plot_lines('smooth loss', loss_smooth.item())
        viz.plot_lines('sparsity loss', loss_sparse.item())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
