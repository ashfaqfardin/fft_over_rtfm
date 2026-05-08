import numpy as np
import torch

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.index[name] = x + 1

    def disp_image(self, name, img):
        pass

    def lines(self, name, line, X=None):
        pass

    def scatter(self, name, data):
        pass


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1]), dtype=np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            idx = min(r[i], len(feat) - 1)
            new_feat[i, :] = feat[idx, :]
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0
    return ret


def save_best_record(test_info, file_path):
    with open(file_path, "w") as fo:
        fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
        fo.write(str(test_info["test_AUC"][-1]))
