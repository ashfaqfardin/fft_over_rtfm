import os
import random
import time

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


def seed_everything(seed=42):
    """
    Makes runs more reproducible.
    Does not remove randomness completely on all hardware,
    but makes ablation comparisons much cleaner.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def parse_active_mods(mod_arg):
    """
    Parses comma-separated modifications like:
        --mod 1,2,3

    Returns:
        set({1, 2, 3})
    """
    active_mods = set()

    if not mod_arg:
        return active_mods

    for m in mod_arg.split(','):
        m = m.strip()

        if not m:
            continue

        if not m.isdigit():
            raise ValueError(f"Invalid mod value: '{m}'. Mods must be integers like 1,2,3.")

        active_mods.add(int(m))

    return active_mods


def build_run_name(args, active_mods):
    """
    Builds a unique model name so ablation runs do not overwrite each other.
    Keeps your existing naming logic.
    """
    if active_mods:
        mod_label = 'mods_' + ''.join(str(m) for m in sorted(active_mods))
    else:
        mod_label = 'baseline'

    run_name = args.model_name

    if active_mods:
        run_name = run_name + '_' + mod_label + '_'

    if args.loss != 'rtfm':
        run_name = run_name + args.loss + '_'

    return run_name, mod_label


def build_dataloaders(args):
    """
    Keeps the same dataloader behaviour as your current code:
    batch size, shuffle, drop_last, test batch_size=1.
    """
    num_workers = getattr(args, "num_workers", 0)

    # pin_memory is only useful when using CUDA.
    pin_memory = torch.cuda.is_available()

    common_train_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    test_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    if num_workers > 0:
        common_train_kwargs["persistent_workers"] = True
        test_kwargs["persistent_workers"] = True

    train_nloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=True),
        **common_train_kwargs
    )

    train_aloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=False),
        **common_train_kwargs
    )

    test_loader = DataLoader(
        Dataset(args, test_mode=True),
        **test_kwargs
    )

    return train_nloader, train_aloader, test_loader


def build_optimizer(model, config, args):
    """
    Keeps your two-group optimizer logic:
    - normal params: base LR, wd=0.005
    - spectral params: LR x 2, wd=1e-3
    """
    spectral_keys = {'W_quant', 'bin_gate'}

    spectral_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(k in name for k in spectral_keys):
            spectral_params.append(param)
        else:
            other_params.append(param)

    if spectral_params:
        optimizer = optim.Adam(
            [
                {
                    'params': other_params,
                    'lr': config.lr[0],
                    'weight_decay': 0.005
                },
                {
                    'params': spectral_params,
                    'lr': config.lr[0] * 2,
                    'weight_decay': 1e-3
                },
            ]
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr[0],
            weight_decay=0.005
        )

    return optimizer


def build_scheduler(optimizer, args):
    """
    Linear warmup then constant LR.
    Same training logic as your current script.
    """
    warmup_epochs = max(1, getattr(args, "warmup_epochs", 1))

    def lr_lambda(step):
        if step < warmup_epochs:
            return float(step + 1) / float(warmup_epochs)
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint_and_curves(
    model,
    args,
    test_info,
    fpr,
    tpr,
    precision,
    recall,
    best_auc,
    step,
    output_path
):
    """
    Saves the exact same items as your original code,
    but with safer paths.
    """
    ckpt_dir = './ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(ckpt_dir, args.model_name + 'best-i3d.pkl')
    )

    save_best_record(
        test_info,
        os.path.join(output_path, args.model_name + 'best-AUC.txt')
    )

    np.save(args.model_name + 'fpr.npy', fpr)
    np.save(args.model_name + 'tpr.npy', tpr)
    np.save(args.model_name + 'precision.npy', precision)
    np.save(args.model_name + 'recall.npy', recall)

    print(f'\n  *** New best AUC: {best_auc:.4f}  (step {step}) ***')


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    seed = getattr(args, "seed", 42)
    seed_everything(seed)

    active_mods = parse_active_mods(args.mod)
    args.model_name, mod_label = build_run_name(args, active_mods)

    print(f'Active modifications: {sorted(active_mods) if active_mods else "none (baseline RTFM)"}')
    print(f'Run label            : {mod_label}')
    print(f'Loss function        : {args.loss}')
    print(f'Model name           : {args.model_name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device         : {device}')

    os.makedirs('./ckpt', exist_ok=True)

    train_nloader, train_aloader, test_loader = build_dataloaders(args)

    model = Model(
        args.feature_size,
        args.batch_size,
        active_mods=active_mods,
        k_ratio=args.k_ratio
    )

    print('\nTrainable parameters:')
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        print(name)
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f'\nTotal parameters     : {total_params:,}')
    print(f'Trainable parameters : {trainable_params:,}\n')

    model = model.to(device)

    optimizer = build_optimizer(model, config, args)
    scheduler = build_scheduler(optimizer, args)

    if args.loss not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{args.loss}'. Available losses: {list(_LOSS_REGISTRY.keys())}"
        )

    loss_fn = _LOSS_REGISTRY[args.loss]().to(device)

    test_info = {
        "epoch": [],
        "test_AUC": [],
        "test_PR_AUC": []
    }

    best_AUC = -1.0
    output_path = getattr(args, "output_path", "")
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # Defaults to testing every epoch, preserving your original behaviour.
    test_interval = getattr(args, "test_interval", 1)

    # Optional AMP support. 
    # If you do not add --amp to option.py, this safely stays False.
    use_amp = bool(getattr(args, "amp", False)) and torch.cuda.is_available()

    print('Running initial evaluation...')
    auc, pr_auc, _, _, _, _, _ = test(test_loader, model, args, viz, device)
    print(f'Initial AUC: {auc:.4f}, Initial PR-AUC: {pr_auc:.4f}')

    start_time = time.time()

    loadern_iter = iter(train_nloader)
    loadera_iter = iter(train_aloader)

    for step in tqdm(
        range(1, args.max_epoch + 1),
        total=args.max_epoch,
        dynamic_ncols=True
    ):
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        pseudo_w = args.pseudo_weight if step > args.pseudo_warmup else 0.0

        train(
            loadern_iter,
            loadera_iter,
            model,
            args.batch_size,
            optimizer,
            viz,
            device,
            loss_fn=loss_fn,
            smooth_weight=args.smooth_weight,
            sparse_weight=args.sparse_weight,
            pseudo_weight=pseudo_w,
            pseudo_threshold=args.pseudo_threshold,
            grad_clip=args.grad_clip
        )

        scheduler.step()

        should_test = (
            step == 1 or
            step % test_interval == 0 or
            step == args.max_epoch
        )

        if should_test:
            auc, pr_auc, fpr, tpr, precision, recall, _ = test(
                test_loader,
                model,
                args,
                viz,
                device
            )

            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_PR_AUC"].append(pr_auc)

            if auc > best_AUC:
                best_AUC = auc

                save_checkpoint_and_curves(
                    model=model,
                    args=args,
                    test_info=test_info,
                    fpr=fpr,
                    tpr=tpr,
                    precision=precision,
                    recall=recall,
                    best_auc=best_AUC,
                    step=step,
                    output_path=output_path
                )

    torch.save(
        model.state_dict(),
        './ckpt/' + args.model_name + 'final.pkl'
    )

    elapsed = time.time() - start_time
    print(f'\nTraining complete.')
    print(f'Best AUC      : {best_AUC:.4f}')
    print(f'Elapsed time  : {elapsed / 60:.2f} minutes')
    print(f'Final model   : ./ckpt/{args.model_name}final.pkl')
