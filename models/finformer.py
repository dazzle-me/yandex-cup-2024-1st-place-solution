
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import timm
from einops import rearrange

from configs.config_0 import CFG

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            margins: dynamic margins
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, margins, s=30.0, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.ls_eps = ls_eps  # label smoothing
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.margins = np.array(margins)
        
    def forward(self, logits, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_features).float()
        # logits = logits.float()
        cosine = F.linear(F.normalize(logits), F.normalize(self.weight))
        
        # cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output
        
class YCModel(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.model_name, pretrained=True, in_chans=1,
            global_pool='', num_classes=0, **cfg.model_kwargs,
        )
        # self.encoder_filters = [f['num_chs'] for f in self.encoder.feature_info]
        # output_dim = 1280
        # output_dim = self.encoder.default_cfg['num_features']
        output_dim = cfg.output_features
        # Print out the output channels
        # print("Number of output channels:", cfg['num_features'])
        if cfg.margins is None:
            raise ValueError("Set margins in cfg")
        self.bn = nn.BatchNorm1d(output_dim)
        self.first_run = True
        self.arcface_head = ArcMarginProduct(
            output_dim, 
            cfg.num_classes,
            cfg.margins,
            s=cfg.arcface_scale, 
            ls_eps=cfg.ls_eps,
        )

    def forward(self, x, labels, mask):
        x = self.extract(x, mask)
        return self.arcface_head(x, labels)

    def extract(self, x, mask):
        x = (x + 20) / 15
        x = x.unsqueeze(1)
        if self.cfg.do_resize:
            x = F.interpolate(x, self.cfg.image_size, mode='bicubic')
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        # print(x.shape)
        # print(x.shape)
        if self.first_run:
            print(f"Feature map shape : {x.shape}")
            self.first_run = False
        if self.cfg.model_name.startswith('eva') or self.cfg.model_name.startswith('beit') or self.cfg.model_name.startswith('coatnet_rmlp_2'):
            x = x.mean(dim=1)
        elif self.cfg.model_name.startswith('swin'):
            x = x.mean(dim=(1, 2))
        else:
            x = x.mean(dim=(2, 3))
        # print(x.shape)
        x = self.bn(x)
        return x