import torch
import torch.nn as nn
import torch.nn.init as torch_init

from new_modules import (
    TemporalDFFN,
    TemporalFSAS,
    FreqGatedClassifier,
    GlanceFocusBlock
)

torch.set_default_dtype(torch.float32)


def weight_init(m):
    """
    Original RTFM-style Xavier init for Conv/Linear layers.
    Keeps behaviour consistent with the current code.
    """
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(m, "weight") and m.weight is not None:
            torch_init.xavier_uniform_(m.weight)

        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


class _NonLocalBlockND(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        sub_sample=True,
        bn_layer=True
    ):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3], "dimension must be 1, 2, or 3"

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = max(1, in_channels // 2)

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                bn(self.in_channels)
            )

            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).reshape(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).reshape(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).reshape(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        # Original code used f / N rather than softmax.
        # Keeping this unchanged for baseline compatibility.
        n = f.size(-1)
        f_div_C = f / n

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        sub_sample=True,
        bn_layer=True
    ):
        super(NONLocalBlock1D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=1,
            sub_sample=sub_sample,
            bn_layer=bn_layer
        )


class Aggregate(nn.Module):
    def __init__(self, len_feature, active_mods=None):
        super(Aggregate, self).__init__()

        if active_mods is None:
            active_mods = set()

        self.active_mods = set(active_mods)
        self.len_feature = len_feature

        bn = nn.BatchNorm1d

        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            bn(512)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dilation=2,
                padding=2
            ),
            nn.ReLU(inplace=True),
            bn(512)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dilation=4,
                padding=4
            ),
            nn.ReLU(inplace=True),
            bn(512)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv1d(
                in_channels=2048,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv1d(
                in_channels=2048,
                out_channels=2048,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048)
        )

        # Mod 1: complex-valued spectral filter on concatenated PDC output.
        if 1 in self.active_mods:
            self.dffn_combined = TemporalDFFN(channels=1536)

        # Mod 2: TemporalFSAS replaces original non-local block.
        if 2 in self.active_mods:
            self.non_local = TemporalFSAS(channels=512, reduction=2)
        else:
            self.non_local = NONLocalBlock1D(
                512,
                sub_sample=False,
                bn_layer=True
            )

        # Mod 5: GlanceFocusBlock after temporal attention.
        if 5 in self.active_mods:
            self.glance_focus = GlanceFocusBlock(channels=512)

    def forward(self, x):
        """
        Input:
            x: (B, T, F)

        Output:
            out: (B, T, F)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Aggregate expected input shape (B, T, F), got {tuple(x.shape)}"
            )

        out = x.permute(0, 2, 1).contiguous()   # (B, F, T)
        residual = out

        out1 = self.conv_1(out)                 # (B, 512, T)
        out2 = self.conv_2(out)                 # (B, 512, T)
        out3 = self.conv_3(out)                 # (B, 512, T)

        out_d = torch.cat((out1, out2, out3), dim=1)  # (B, 1536, T)

        if 1 in self.active_mods:
            out_d = self.dffn_combined(out_d)

        out = self.conv_4(out)                  # (B, 512, T)
        out = self.non_local(out)               # (B, 512, T)

        if 5 in self.active_mods:
            out = self.glance_focus(out)

        out = torch.cat((out_d, out), dim=1)    # (B, 2048, T)
        out = self.conv_5(out)                  # (B, 2048, T)

        out = out + residual
        out = out.permute(0, 2, 1).contiguous() # (B, T, F)

        return out


class Model(nn.Module):
    def __init__(self, n_features, batch_size, active_mods=None, k_ratio=0.1):
        super(Model, self).__init__()

        if active_mods is None:
            active_mods = set()

        self.active_mods = set(active_mods)
        self.batch_size = batch_size
        self.num_segments = 32

        # k_ratio controls how many snippets are selected per bag.
        self.k_abn = max(1, int(self.num_segments * k_ratio))
        self.k_nor = max(1, int(self.num_segments * k_ratio))

        self.Aggregate = Aggregate(
            len_feature=2048,
            active_mods=self.active_mods
        )

        # Mod 4: frequency-gated classifier replaces original FC score head.
        if 4 in self.active_mods:
            self.classifier = FreqGatedClassifier(
                n_features=n_features,
                hidden=512,
                dropout=0.7
            )

        # Original RTFM scoring head.
        # Kept even when Mod 4 is active so checkpoint structure remains stable.
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weight_init)

        # Re-zero TemporalFSAS output projection after global weight init.
        # This keeps the residual branch initially identity-like.
        if 2 in self.active_mods:
            self._rezero_temporal_fsas()

    def _rezero_temporal_fsas(self):
        for module in self.modules():
            if isinstance(module, TemporalFSAS):
                if hasattr(module, "W") and isinstance(module.W, nn.Sequential):
                    if len(module.W) > 1:
                        if hasattr(module.W[1], "weight"):
                            nn.init.constant_(module.W[1].weight, 0)
                        if hasattr(module.W[1], "bias"):
                            nn.init.constant_(module.W[1].bias, 0)

    def _score_features(self, features):
        """
        Input:
            features: (B * ncrops, T, F)

        Output:
            scores: (B * ncrops, T, 1)
        """
        if 4 in self.active_mods:
            return self.classifier(features)

        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)

        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)

        scores = self.sigmoid(self.fc3(scores))

        return scores

    @staticmethod
    def _select_topk_features(features, idx):
        """
        Vectorised replacement for the old loop + repeated torch.cat.

        Original logic:
            features: (N * ncrops, T, F)
            view to:  (N, ncrops, T, F)
            permute:  (ncrops, N, T, F)
            gather top-k snippets per crop
            output:   (ncrops * N, k, F)

        Args:
            features: (N * ncrops, T, F)
            idx:      (N, k)

        Returns:
            selected: (ncrops * N, k, F)
        """
        n_bags = idx.size(0)
        k = idx.size(1)

        total_crop_bags, t, f = features.shape

        if n_bags <= 0:
            raise ValueError("n_bags must be positive in _select_topk_features")

        if total_crop_bags % n_bags != 0:
            raise ValueError(
                "Feature count is not divisible by number of bags: "
                f"features={total_crop_bags}, bags={n_bags}"
            )

        ncrops = total_crop_bags // n_bags

        features = features.reshape(n_bags, ncrops, t, f)
        features = features.permute(1, 0, 2, 3).contiguous()  # (C, N, T, F)

        gather_idx = idx.unsqueeze(0).unsqueeze(-1)           # (1, N, k, 1)
        gather_idx = gather_idx.expand(ncrops, n_bags, k, f)  # (C, N, k, F)

        selected = torch.gather(features, dim=2, index=gather_idx)
        selected = selected.reshape(ncrops * n_bags, k, f)

        return selected

    @staticmethod
    def _select_topk_scores(scores, idx):
        """
        Args:
            scores: (N, T, 1)
            idx:    (N, k)

        Returns:
            score_selected: (N, 1)
        """
        gather_idx = idx.unsqueeze(2).expand(-1, -1, scores.shape[2])
        selected_scores = torch.gather(scores, dim=1, index=gather_idx)
        selected_scores = torch.mean(selected_scores, dim=1)

        return selected_scores

    def forward(self, inputs):
        """
        Expected input:
            inputs: (B, ncrops, T, F)

        Returns same tuple structure as original code.
        """
        if inputs.dim() != 4:
            raise ValueError(
                f"Model expected input shape (B, ncrops, T, F), got {tuple(inputs.shape)}"
            )

        k_abn = self.k_abn
        k_nor = self.k_nor

        bs, ncrops, t, f = inputs.size()

        # Safety: top-k cannot exceed temporal length.
        k_abn = min(k_abn, t)
        k_nor = min(k_nor, t)

        out = inputs.reshape(-1, t, f)       # (B * ncrops, T, F)

        out = self.Aggregate(out)
        out = self.drop_out(out)

        features = out                       # (B * ncrops, T, F)

        scores = self._score_features(features)       # (B * ncrops, T, 1)
        scores = scores.reshape(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)               # (B, T, 1)

        # Mod 3: deviation features for both selection and loss gathering.
        if 3 in self.active_mods:
            feat_baseline = features.mean(dim=1, keepdim=True)
            loss_features = features - feat_baseline
            feat_magnitudes = loss_features.norm(p=2, dim=2)
        else:
            loss_features = features
            feat_magnitudes = features.norm(p=2, dim=2)

        # Convert crop-level magnitudes to video-level magnitudes.
        feat_magnitudes = feat_magnitudes.reshape(bs, ncrops, -1).mean(1)

        # During training, the input batch is expected to be:
        # first self.batch_size normal videos, then self.batch_size abnormal videos.
        #
        # During inference, batch size is usually 1, so the abnormal branch is
        # mirrored from the normal branch, exactly like the original code.
        normal_bags = min(self.batch_size, bs)

        normal_crop_count = normal_bags * ncrops

        normal_features = loss_features[:normal_crop_count]
        normal_scores = scores[:normal_bags]
        nfea_magnitudes = feat_magnitudes[:normal_bags]

        abnormal_features = loss_features[normal_crop_count:]
        abnormal_scores = scores[normal_bags:]
        afea_magnitudes = feat_magnitudes[normal_bags:]

        n_size = nfea_magnitudes.shape[0]

        # Inference path: only one video exists.
        # Mirror normal tensors into abnormal tensors to preserve output structure.
        if nfea_magnitudes.shape[0] == 1 and afea_magnitudes.shape[0] == 0:
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        # Extra safety for unusual small batches.
        if afea_magnitudes.shape[0] == 0:
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        ####### abnormal videos — select top-k by feature magnitude #######
        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)

        afea_magnitudes_drop = afea_magnitudes * select_idx

        idx_abn = torch.topk(
            afea_magnitudes_drop,
            k=k_abn,
            dim=1
        )[1]

        total_select_abn_feature = self._select_topk_features(
            abnormal_features,
            idx_abn
        )

        score_abnormal = self._select_topk_scores(
            abnormal_scores,
            idx_abn
        )

        ####### normal videos — select top-k by feature magnitude #######
        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)

        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal

        idx_normal = torch.topk(
            nfea_magnitudes_drop,
            k=k_nor,
            dim=1
        )[1]

        total_select_nor_feature = self._select_topk_features(
            normal_features,
            idx_normal
        )

        score_normal = self._select_topk_scores(
            normal_scores,
            idx_normal
        )

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return (
            score_abnormal,
            score_normal,
            feat_select_abn,
            feat_select_normal,
            feat_select_abn,
            feat_select_abn,
            scores,
            feat_select_abn,
            feat_select_abn,
            feat_magnitudes
        )
