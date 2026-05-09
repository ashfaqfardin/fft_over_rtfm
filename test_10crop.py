import time
import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score


def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred             = torch.zeros(0, device=device)
        per_video_scores = []   # (T,) segment-level scores per video
        t_start          = time.time()

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_select_normal_bottom, logits, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes \
                = model(inputs=input)

            logits = torch.squeeze(logits, -1)   # (bs, T, 1) → (bs, T)
            logits = torch.mean(logits, 0)        # mean over batch dim → (T,)
            sig    = logits.flatten()             # guarantee 1-D
            per_video_scores.append(sig.cpu().detach().numpy())
            pred = torch.cat((pred, sig))

        n_videos      = len(per_video_scores)
        ms_per_video  = 1000.0 * (time.time() - t_start) / max(n_videos, 1)

        gt      = np.load(args.gt)
        pred_np = np.repeat(pred.cpu().detach().numpy(), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred_np)
        rec_auc  = auc(fpr, tpr)

        # average_precision_score = interpolated left-Riemann sum (standard for VAD papers)
        pr_auc    = average_precision_score(list(gt), pred_np)
        precision, recall, th = precision_recall_curve(list(gt), pred_np)

        print(f'auc : {rec_auc:.4f}  |  pr_auc : {pr_auc:.4f}  |  {ms_per_video:.1f} ms/video')

        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc',    rec_auc)
        viz.lines('scores', pred_np)
        viz.lines('roc',    tpr, fpr)
        return rec_auc, pr_auc, fpr, tpr, precision, recall, per_video_scores


def collect_embeddings(dataloader, model, device):
    """Extract per-video Aggregate-layer embeddings via a forward hook.

    Used for UMAP / t-SNE visualisation.

    Returns
    -------
    embeddings      : np.ndarray  (N_videos, F_agg)  — mean-pooled over crops & time
    per_video_scores: list of np.ndarray (T_i,)      — segment-level anomaly scores
    """
    embeddings       = []
    per_video_scores = []
    feats_buf        = []

    def _hook(module, inp, out):
        feats_buf.append(out.detach().cpu())

    handle = model.Aggregate.register_forward_hook(_hook)

    with torch.no_grad():
        model.eval()
        for input in dataloader:
            feats_buf.clear()
            input           = input.to(device)
            input           = input.permute(0, 2, 1, 3)   # (1, ncrops, T, F)

            _, _, _, _, _, _, logits, _, _, _ = model(inputs=input)
            logits = torch.squeeze(logits, -1).mean(0).flatten()  # (T,)
            per_video_scores.append(logits.cpu().numpy())

            # feats_buf[0]: (bs*ncrops, T, F_agg) — mean over crops and time → (F_agg,)
            feat = feats_buf[0].mean(0).mean(0).numpy()
            embeddings.append(feat)

    handle.remove()
    return np.array(embeddings), per_video_scores
