import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_dtype(torch.float32)

from new_modules import TemporalDFFN, TemporalFSAS, freq_magnitude, FreqGatedClassifier, GlanceFocusBlock


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

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
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Aggregate(nn.Module):
    def __init__(self, len_feature, active_mods=None):
        super(Aggregate, self).__init__()
        if active_mods is None:
            active_mods = set()
        self.active_mods = active_mods

        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
        )

        # Mod 1: T-DFFN frequency gating after each PDC branch
        if 1 in active_mods:
            self.dffn1 = TemporalDFFN(channels=512, patch_size=4)
            self.dffn2 = TemporalDFFN(channels=512, patch_size=4)
            self.dffn3 = TemporalDFFN(channels=512, patch_size=4)

        # Mod 2: TemporalFSAS replaces NONLocalBlock1D
        if 2 in active_mods:
            self.non_local = TemporalFSAS(channels=512, reduction=2)
        else:
            self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)

        # Mod 5: GlanceFocusBlock after non-local attention (MGFN, AAAI 2023)
        if 5 in active_mods:
            self.glance_focus = GlanceFocusBlock(channels=512)

    def forward(self, x):
        out = x.permute(0, 2, 1)   # (B, T, F) → (B, F, T)
        residual = out

        out1 = self.conv_1(out)    # (B, 512, T)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out)

        # Mod 1: apply T-DFFN frequency gating to each PDC branch output
        if 1 in self.active_mods:
            out1 = self.dffn1(out1)
            out2 = self.dffn2(out2)
            out3 = self.dffn3(out3)

        out_d = torch.cat((out1, out2, out3), dim=1)   # (B, 1536, T)
        out = self.conv_4(out)                          # (B, 512, T)
        out = self.non_local(out)                       # (B, 512, T)

        # Mod 5: Glance-and-Focus channel+local attention (MGFN, AAAI 2023)
        if 5 in self.active_mods:
            out = self.glance_focus(out)                # (B, 512, T)

        out = torch.cat((out_d, out), dim=1)            # (B, 2048, T)
        out = self.conv_5(out)                          # (B, 2048, T)
        out = out + residual
        out = out.permute(0, 2, 1)                      # (B, T, F)
        return out


class Model(nn.Module):
    def __init__(self, n_features, batch_size, active_mods=None, k_ratio=0.1):
        super(Model, self).__init__()
        if active_mods is None:
            active_mods = set()
        self.active_mods = active_mods
        self.batch_size = batch_size
        self.num_segments = 32
        # k_ratio controls how many snippets are selected per bag (Improvement 4)
        self.k_abn = max(1, int(self.num_segments * k_ratio))
        self.k_nor = max(1, int(self.num_segments * k_ratio))

        self.Aggregate = Aggregate(len_feature=2048, active_mods=active_mods)

        # Mod 4: FreqGatedClassifier replaces the fc1/fc2/fc3 scoring head
        if 4 in active_mods:
            self.classifier = FreqGatedClassifier(
                n_features=n_features, hidden=512, dropout=0.7
            )

        # Keep original FC layers; used by baseline and also by weight_init sweep
        self.fc1      = nn.Linear(n_features, 512)
        self.fc2      = nn.Linear(512, 128)
        self.fc3      = nn.Linear(128, 1)
        self.drop_out = nn.Dropout(0.7)
        self.relu     = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()
        self.apply(weight_init)

        # Re-zero TemporalDFFN out_proj after weight_init overwrites it with xavier.
        # Guarantees exact identity residual at step 0 (same pattern as TemporalFSAS).
        if 1 in active_mods:
            for module in self.modules():
                if isinstance(module, TemporalDFFN):
                    nn.init.zeros_(module.out_proj.weight)

        # Re-zero TemporalFSAS BN output projections after weight_init runs
        # (weight_init applies xavier to the preceding Conv1d but leaves BN
        # weights alone; this block re-confirms the zero-init is intact)
        if 2 in active_mods:
            for module in self.modules():
                if isinstance(module, TemporalFSAS):
                    nn.init.constant_(module.W[1].weight, 0)
                    nn.init.constant_(module.W[1].bias,   0)

    def forward(self, inputs):
        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)
        out = self.Aggregate(out)
        out = self.drop_out(out)

        features = out   # (B*ncrops, T, F)

        # Mod 4: FreqGatedClassifier; otherwise original fc head
        if 4 in self.active_mods:
            scores = self.classifier(features)          # (B*ncrops, T, 1)
        else:
            scores = self.relu(self.fc1(features))
            scores = self.drop_out(scores)
            scores = self.relu(self.fc2(scores))
            scores = self.drop_out(scores)
            scores = self.sigmoid(self.fc3(scores))

        scores = scores.view(bs, ncrops, -1).mean(1)   # (B, T)
        scores = scores.unsqueeze(dim=2)               # (B, T, 1)

        normal_features   = features[0:self.batch_size * 10]
        normal_scores     = scores[0:self.batch_size]
        abnormal_features = features[self.batch_size * 10:]
        abnormal_scores   = scores[self.batch_size:]

        # Mod 3: frequency-domain feature magnitude; otherwise standard L2 norm
        if 3 in self.active_mods:
            feat_magnitudes = freq_magnitude(features)
        else:
            feat_magnitudes = torch.norm(features, p=2, dim=2)

        feat_magnitudes  = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes  = feat_magnitudes[0:self.batch_size]
        afea_magnitudes  = feat_magnitudes[self.batch_size:]
        n_size           = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:   # inference: batch size is 1
            afea_magnitudes   = nfea_magnitudes
            abnormal_scores   = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)

        #######  abnormal videos — select top-k by feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn      = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score  = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)

        #######  normal videos — select top-k by feature magnitude  #######
        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal      = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0, device=inputs.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal     = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

        feat_select_abn    = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return (score_abnormal, score_normal,
                feat_select_abn, feat_select_normal,
                feat_select_abn, feat_select_abn,
                scores,
                feat_select_abn, feat_select_abn,
                feat_magnitudes)
