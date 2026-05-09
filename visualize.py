#!/usr/bin/env python3
"""
visualize.py  —  Anomaly score + feature magnitude curves for selected test videos.

Reproduces the dual-axis plots from the RTFM paper:
  blue  line  (left  axis) = anomaly score per segment
  orange line (right axis) = L2-norm / frequency magnitude per segment
  pink shading             = ground-truth anomalous frames

Usage examples
--------------
# By video index (0-based position in the test list):
python visualize.py --ckpt ckpt/rtfm_ucf_best-i3d.pkl --videos 0,5,50,100

# By name substring (case-insensitive):
python visualize.py --ckpt ckpt/rtfm_ucf_best-i3d.pkl --names Stealing,Robbery,Normal

# With FFTFormer mods (must match the training run):
python visualize.py --ckpt ckpt/rtfm_ucf_mods_1234_best-i3d.pkl --mod 1,2,3,4 --names Shoplifting

# Custom output file and layout:
python visualize.py --ckpt ckpt/rtfm_ucf_best-i3d.pkl --videos 0,1,2,3,4,5 --cols 3 --output curves.png
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import Model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='RTFM score curve visualiser')
    p.add_argument('--ckpt',          required=True, help='checkpoint .pkl path')
    p.add_argument('--mod',           default='',    help='comma-separated mods used at training time')
    p.add_argument('--videos',        default='',    help='comma-separated 0-based video indices')
    p.add_argument('--names',         default='',    help='comma-separated filename substrings to match')
    p.add_argument('--output',        default='score_curves.png')
    p.add_argument('--cols',          type=int, default=3, help='subplots per row (default 3)')
    p.add_argument('--feature-size',  type=int, default=2048)
    p.add_argument('--batch-size',    type=int, default=32)
    p.add_argument('--test-rgb-list', default='list/ucf-i3d-test.list')
    p.add_argument('--gt',            default='list/gt-ucf-local.npy')
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, feature_size, batch_size, active_mods, device):
    model = Model(feature_size, batch_size, active_mods=active_mods)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_video(feat_path, model, device):
    """Return (scores, magnitudes) both shape (T,) for one video."""
    feats = np.load(feat_path, allow_pickle=True).astype(np.float32)
    # feats: (T, ncrops, F)
    inp = torch.from_numpy(feats).unsqueeze(0)   # (1, T, ncrops, F)
    inp = inp.permute(0, 2, 1, 3).to(device)     # (1, ncrops, T, F)

    with torch.no_grad():
        _, _, _, _, _, _, logits, _, _, feat_magnitudes = model(inputs=inp)

    # logits:          (1, T, 1)  → squeeze last dim → mean over ncrops dim → (T,)
    scores = logits.squeeze(-1).mean(0).flatten().cpu().numpy()

    # feat_magnitudes: (1, T)  already averaged over ncrops inside model.forward
    mags   = feat_magnitudes.squeeze(0).cpu().numpy()

    return scores, mags


# ── GT helpers ────────────────────────────────────────────────────────────────

def build_video_index(test_list):
    """
    Precompute (path, frame_offset, n_frames) for every video.
    Loads each .npy file once to count segments.
    """
    index  = []
    offset = 0
    for line in test_list:
        path = line.strip().split()[0]
        try:
            feats    = np.load(path, allow_pickle=True)
            n_frames = feats.shape[0] * 16      # segments × 16 frames/segment
        except Exception:
            n_frames = 0
        index.append((path, offset, n_frames))
        offset += n_frames
    return index


def anomaly_regions(gt_frames):
    """Return list of (start, end) frame pairs where GT == 1."""
    regions, in_anom, start = [], False, 0
    for i, v in enumerate(gt_frames):
        if v == 1 and not in_anom:
            start, in_anom = i, True
        elif v == 0 and in_anom:
            regions.append((start, i))
            in_anom = False
    if in_anom:
        regions.append((start, len(gt_frames)))
    return regions


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_video(ax, scores, mags, gt_frames, title):
    frames_per_seg = 16
    n_seg          = len(scores)
    # x positions = centre frame of each segment
    x = np.arange(n_seg) * frames_per_seg + frames_per_seg // 2

    # Pink GT shading
    for (s, e) in anomaly_regions(gt_frames):
        ax.axvspan(s, e, alpha=0.3, color='pink', zorder=0)

    ax2 = ax.twinx()
    ax.plot(x,  scores, color='blue',   linewidth=1.2, zorder=2)
    ax2.plot(x, mags,   color='orange', linewidth=1.2, zorder=2)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Frame Number',   fontsize=8)
    ax.set_ylabel('Anomaly Score',  fontsize=7, color='blue')
    ax2.set_ylabel('L2-Norm',       fontsize=7, color='orange')
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, pad=3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse active mods
    active_mods = set()
    if args.mod:
        for m in args.mod.split(','):
            m = m.strip()
            if m.isdigit():
                active_mods.add(int(m))

    model     = load_model(args.ckpt, args.feature_size, args.batch_size, active_mods, device)
    test_list = list(open(args.test_rgb_list))
    gt_flat   = np.load(args.gt)

    # Resolve video indices
    indices = []
    if args.videos:
        indices += [int(x.strip()) for x in args.videos.split(',') if x.strip().isdigit()]
    if args.names:
        substrings = [s.strip().lower() for s in args.names.split(',')]
        for i, line in enumerate(test_list):
            name = os.path.basename(line.strip().split()[0]).lower()
            if any(s in name for s in substrings):
                indices.append(i)
    if not indices:
        print('No videos selected. Use --videos 0,1,2 or --names Stealing,Normal')
        return

    indices = sorted(set(indices))

    # Precompute GT offsets (loads all .npy headers once)
    print('Building video index...')
    vid_index = build_video_index(test_list)

    # Layout
    n    = len(indices)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows),
                             squeeze=False)

    for plot_i, vid_idx in enumerate(indices):
        row, col = divmod(plot_i, cols)
        ax       = axes[row][col]

        path, offset, n_frames = vid_index[vid_idx]
        name = os.path.splitext(os.path.basename(path))[0].replace('_x264_i3d', '')

        print(f'  [{plot_i+1}/{n}]  {name}')
        try:
            scores, mags = infer_video(path, model, device)
        except Exception as e:
            print(f'    ERROR: {e}')
            ax.set_visible(False)
            continue

        gt_frames = gt_flat[offset: offset + n_frames] if n_frames > 0 else np.zeros(len(scores) * 16)
        plot_video(ax, scores, mags, gt_frames, name)

    # Hide unused subplot panels
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].set_visible(False)

    # Shared legend
    handles = [
        mpatches.Patch(color='blue',   label='Anomaly Score'),
        mpatches.Patch(color='orange', label='L2-Norm / Freq-Mag'),
        mpatches.Patch(color='pink',   label='GT Anomaly', alpha=0.6),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0))

    plt.suptitle('Anomaly Score Curves', fontsize=12, y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {args.output}')


if __name__ == '__main__':
    main()
