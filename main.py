from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train, _LOSS_REGISTRY
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *

viz = Visualizer(env='ucf crime 10 crop', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    # Parse comma-separated mod list (e.g. "1,2,3") into a set of ints
    active_mods = set()
    if args.mod:
        for m in args.mod.split(','):
            m = m.strip()
            if m.isdigit():
                active_mods.add(int(m))

    mod_label = ('mods_' + ''.join(str(m) for m in sorted(active_mods))
                 if active_mods else 'baseline')
    print(f'Active modifications: {sorted(active_mods) if active_mods else "none (baseline RTFM)"}')

    # Append mod + loss tags so ablation runs don't overwrite each other
    if active_mods:
        args.model_name = args.model_name + '_' + mod_label + '_'
    if args.loss != 'rtfm':
        args.model_name = args.model_name + args.loss + '_'
    print(f'Loss function : {args.loss}')

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size,
                  active_mods=active_mods, k_ratio=args.k_ratio)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    # Spectral parameters (W_quant + bin_gate inside TemporalDFFN) get:
    #   lr×2  — spectral weights benefit from faster updates
    #   wd=1e-3 — pulls toward identity (0), prevents spectral overfitting
    # dyn_gate (a Linear layer) stays in other_params: standard lr, wd=0.005.
    _spectral_keys = {'W_quant', 'bin_gate'}
    w_quant_params = [p for n, p in model.named_parameters()
                      if any(k in n for k in _spectral_keys)]
    other_params   = [p for n, p in model.named_parameters()
                      if not any(k in n for k in _spectral_keys)]

    if w_quant_params:
        optimizer = optim.Adam([
            {'params': other_params,   'lr': config.lr[0], 'weight_decay': 0.005},
            {'params': w_quant_params, 'lr': config.lr[0] * 2, 'weight_decay': 1e-3},
        ])
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=config.lr[0], weight_decay=0.005)

    # Linear warmup then fixed LR — matches RTFM / MGFN / MIST convention.
    # These papers all use Adam with a constant LR on pre-extracted I3D features;
    # cosine decay is unnecessary and drives the LR to zero before training ends.
    def lr_lambda(step):
        if step < args.warmup_epochs:
            return (step + 1) / args.warmup_epochs   # 0 → 1 linearly
        return 1.0                                    # hold at base LR

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Build loss function once — not per-step — so any future stateful losses
    # retain their parameters across training steps.
    loss_fn = _LOSS_REGISTRY[args.loss]().to(device)

    test_info = {"epoch": [], "test_AUC": [], "test_PR_AUC": []}
    best_AUC = -1
    output_path = ''   # put your own path here
    auc, pr_auc, _, _, _, _, _ = test(test_loader, model, args, viz, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        # Pseudo-label loss activates only after warmup epochs (MIST, CVPR 2021)
        pseudo_w = args.pseudo_weight if step > args.pseudo_warmup else 0.0
        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device,
              loss_fn=loss_fn,
              smooth_weight=args.smooth_weight,
              sparse_weight=args.sparse_weight,
              pseudo_weight=pseudo_w,
              pseudo_threshold=args.pseudo_threshold,
              grad_clip=args.grad_clip)

        scheduler.step()

        auc, pr_auc, fpr, tpr, precision, recall, _ = test(test_loader, model, args, viz, device)
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR_AUC"].append(pr_auc)

        if auc > best_AUC:
            best_AUC = auc
            torch.save(model.state_dict(),
                       './ckpt/' + args.model_name + 'best-i3d.pkl')
            save_best_record(test_info,
                             os.path.join(output_path, args.model_name + 'best-AUC.txt'))
            np.save(args.model_name + 'fpr.npy', fpr)
            np.save(args.model_name + 'tpr.npy', tpr)
            np.save(args.model_name + 'precision.npy', precision)
            np.save(args.model_name + 'recall.npy', recall)
            print(f'\n  *** New best AUC: {best_AUC:.4f}  (step {step}) ***')

    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
