from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
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

    # Append mod tag to checkpoint name so ablation runs don't overwrite each other
    if active_mods:
        args.model_name = args.model_name + '_' + mod_label + '_'

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size, active_mods=active_mods)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    # W_quant params (inside TemporalDFFN) must not be regularised toward zero.
    # Give them zero weight decay and a slightly higher lr.
    w_quant_params = [p for n, p in model.named_parameters() if 'W_quant' in n]
    other_params   = [p for n, p in model.named_parameters() if 'W_quant' not in n]

    if w_quant_params:
        optimizer = optim.Adam([
            {'params': other_params,   'lr': config.lr[0], 'weight_decay': 0.005},
            {'params': w_quant_params, 'lr': config.lr[0] * 2, 'weight_decay': 0.0},
        ])
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = ''   # put your own path here
    auc, _, _, _, _ = test(test_loader, model, args, viz, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        auc, fpr, tpr, precision, recall = test(test_loader, model, args, viz, device)
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)

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
