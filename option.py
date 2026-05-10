import argparse

parser = argparse.ArgumentParser(description='RTFM')

parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB, AUDIO, or MIX')

# Use UCF-Crime first
parser.add_argument('--rgb-list', default='list/ucf-i3d.list', help='list of training rgb features')
parser.add_argument('--test-rgb-list', default='list/ucf-i3d-test.list', help='list of test rgb features')
parser.add_argument('--gt', default='list/gt-ucf-local.npy', help='file of ground truth')

# Mac-safe / CPU-safe defaults
parser.add_argument('--gpus', default=0, type=int, help='number of gpus to use')
parser.add_argument('--lr', type=str, default='[0.001]*100', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data')
parser.add_argument('--workers', default=0, type=int, help='number of workers in dataloader')

parser.add_argument('--model-name', default='rtfm_ucf', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of classes')
parser.add_argument('--dataset', default='ucf', help='dataset to train on')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting')
parser.add_argument('--max-epoch', type=int, default=100, help='maximum iteration to train')
parser.add_argument('--mod', default='', type=str,
                    help='comma-separated mods to enable: '
                         '1=T-DFFN PDC branches, 2=T-FSAS attention, '
                         '3=freq magnitude, 4=FreqGated classifier head, '
                         '5=GlanceFocus (MGFN AAAI-23) '
                         '(e.g. --mod 1,2 or --mod 1,2,3,4,5)')
parser.add_argument('--loss', default='rtfm',
                    choices=['rtfm', 'ranking', 'focal', 'contrastive', 'mgfn'],
                    help='loss function: rtfm (default) | ranking (Sultani MIL hinge) | '
                         'focal (Focal-BCE) | contrastive (cosine-margin) | '
                         'mgfn (magnitude contrastive, AAAI-23)')
parser.add_argument('--smooth-weight', type=float, default=8e-4,
                    help='temporal smoothness regulariser weight (default: 8e-4)')
parser.add_argument('--sparse-weight', type=float, default=8e-3,
                    help='sparsity regulariser weight (default: 8e-3)')
parser.add_argument('--k-ratio', type=float, default=0.1,
                    help='fraction of T snippets selected per bag for MIL top-k '
                         '(default: 0.1 = 3 out of 32; e.g. 0.15 → 4, 0.2 → 6)')
parser.add_argument('--pseudo-weight', type=float, default=0.0,
                    help='weight for pseudo-label self-training loss (MIST, CVPR-21); '
                         '0 disables it (default: 0.0, try 0.05–0.1)')
parser.add_argument('--pseudo-warmup', type=int, default=20,
                    help='epoch after which pseudo-label loss activates (default: 20)')
parser.add_argument('--pseudo-threshold', type=float, default=0.8,
                    help='confidence threshold for pseudo-label selection (default: 0.8)')
parser.add_argument('--grad-clip', type=float, default=10.0,
                    help='max gradient norm for clipping (default: 10.0; 0 disables clipping)')
parser.add_argument('--warmup-epochs', type=int, default=5,
                    help='number of linear LR warm-up epochs before cosine schedule (default: 5)')