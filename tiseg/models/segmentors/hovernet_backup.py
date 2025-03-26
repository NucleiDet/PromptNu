"""
Modified from vqdang code at https://github.com/vqdang/hover_net/blob/conic/models/hovernet/net_desc.py
"""

import math
from collections import OrderedDict
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from torchvision.transforms import ToPILImage, ToTensor
from typing import Type, Any, Callable, Union, List, Optional

from torch import Tensor
from tiseg.utils import resize
from .base import BaseSegmentor
from ..builder import SEGMENTORS
from ..losses import GradientMSELoss, BatchMultiClassDiceLoss, mdice, tdice
from clip import clip
from PIL import Image
import scipy.signal
from coop import coop
from plip.plip import PLIP
from clip_adapter import clip_adapter


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ResNetExt(ResNet):
    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            self.attnpool = AttentionPool2d(1024 // 32, 2048, 32, 1024)
            self.attnpool.to(x.device)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  #
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
                x5 = self.attnpool(x4)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
            return x1, x2, x3, x4

        return x1, x2, x3, x4, x5

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def resnet50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        model.conv1 = nn.Conv2d(num_input_channels, 64, 7, stride=1, padding=3)

        # model.conv1 = nn.Conv2d(num_input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # model.bn1 = nn.BatchNorm2d(32)

        def load_pretrained(pretrained):
            scripted_model = torch.jit.load(pretrained)
            state_dict = scripted_model.state_dict()

            # 要删除的键的列表
            keys_to_remove = [
                "token_embedding.weight",
                "ln_final.weight",
                "ln_final.bias",
                "positional_embedding",
                "text_projection",
                "logit_scale",
                "input_resolution",
                "context_length",
                "vocab_size",
                "visual.attnpool.positional_embedding"
            ]
            # 用于匹配通配符键的模式
            pattern = re.compile(r"transformer\.resblocks.*")

            # 创建一个新的 state_dict
            new_state_dict = {}

            for key, value in state_dict.items():
                # 检查键是否在删除列表中，或者是否匹配模式
                if key not in keys_to_remove and not pattern.match(key):
                    # 重命名规则：删除前缀 "visual."
                    new_key = key.replace("visual.", "")
                    # 将重命名后的键值对添加到新的 state_dict
                    new_state_dict[new_key] = value
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
            return pretrained_dict

        if pretrained is not None:
            # pretrained = torch.load(pretrained)
            pretrained = load_pretrained(pretrained)
            (missing_keys, unexpected_keys) = model.load_state_dict(pretrained, strict=False)
        return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CLIPResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.c1 = nn.Conv2d(3, 64, 7, stride=1, padding=3)
        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        self.attnpool_clip = AttentionPool2d(input_resolution // 8, 2048, 32, 512)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        # x = stem(x)
        x = self.c1(x)
        x1 = x = self.layer1(x)
        x2 = x = self.layer2(x)
        x3 = x = self.layer3(x)
        x4 = x = self.layer4(x)
        ###attnpool for clip###
        # x5  = self.attnpool_clip(x)
        center_features = []
        sizes = [7, 12, 17, 22, 27, 32]
        for size in sizes:
            start = (32 - size) // 2
            center_feature = x[:, :, start:start + size, start:start + size]
            center_feature = nn.functional.interpolate(center_feature, size=(32, 32), mode='bilinear',
                                                       align_corners=False)
            center_feature = self.attnpool_clip(center_feature)
            center_features.append(center_feature)

        return x1, x2, x3, x4, center_features


class DenseBlock(nn.Module):
    """Dense Block as defined in:
    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger.
    "Densely connected convolutional networks." In Proceedings of the IEEE conference
    on computer vision and pattern recognition, pp. 4700-4708. 2017.
    Only performs `valid` convolution.
    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super().__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        pad_vals = [v // 2 for v in unit_ksize]
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(unit_in_ch, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_in_ch,
                        unit_ch[0],
                        unit_ksize[0],
                        stride=1,
                        padding=pad_vals[0],
                        bias=False,
                    ),
                    nn.BatchNorm2d(unit_ch[0], eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_ch[0],
                        unit_ch[1],
                        unit_ksize[1],
                        stride=1,
                        padding=pad_vals[1],
                        bias=False,
                        groups=split,
                    ),
                ))
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(nn.BatchNorm2d(unit_in_ch, eps=1e-5), nn.ReLU(inplace=True))

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat


class UpSample2x(nn.Module):
    """A layer to scale input by a factor of 2.
    This layer uses Kronecker product underneath rather than the default
    pytorch interpolation.
    """

    def __init__(self):
        super().__init__()
        # correct way to create constant within module
        self.register_buffer("unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32")))
        self.unpool_mat.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """Logic for using layers defined in init.
        Args:
            x (torch.Tensor): Input images, the tensor is in the shape of NCHW.
        Returns:
            ret (torch.Tensor): Input images upsampled by a factor of 2
                via nearest neighbour interpolation. The tensor is the shape
                as NCHW.
        """
        input_shape = list(x.shape)
        # un-squeeze is the same as expand_dims
        # permute is the same as transpose
        # view is the same as reshape
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


@SEGMENTORS.register_module()
class HoverNet(BaseSegmentor):
    """Initialise HoVer-Net."""

    def __init__(self, num_classes, train_cfg, test_cfg, gpu_ids):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.CLASSES = {
            "background",
            "miscellaneous",
            "inflammatory",
            "healthy epithelial",
            "dysplastic or malignant epithelial",
            "fibroblast",
            "muscle",
            "endothelial"
        }

        ###hovernet backbone###
        # self.backbone = ResNetExt.resnet50(3, pretrained=None)
        ###clip backbone###
        pretrained_model = "/root/autodl-tmp/rubyyao/PycharmProjects/nuclei_ins_seg/code/pretrained/RN50.pt"
        self.backbone = CLIPResNet([3, 4, 6, 3], output_dim=512, input_resolution=256, width=64,
                                   pretrained=pretrained_model)
        self.backbone.init_weights()
        self.clip_model, self.clip_preprocess = clip.load(pretrained_model, gpu_ids[0])
        self.logit_scale_num = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_nc = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_tc = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_sc = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.nc_conv = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1)
        self.tc_conv = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1)
        ####

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        ksize = 3
        self.decoder = nn.ModuleDict(
            OrderedDict([
                # ("tp", self.create_decoder_branch(ksize=ksize, out_ch=8)),
                ("tc", self.create_decoder_branch(ksize=ksize, out_ch=1024)),
                # ("np", self.create_decoder_branch(ksize=ksize, out_ch=2)),
                ("nc", self.create_decoder_branch(ksize=ksize, out_ch=1024)),
                ("sc", self.create_decoder_branch(ksize=ksize, out_ch=1024)),
                ("hv", self.create_decoder_branch(ksize=ksize, out_ch=2)),
            ]))

        self.upsample2x = UpSample2x()
        ###coop###
        print("Building custom CLIP")
        self.clip_model = coop.load_clip_to_cpu(pretrained_model)
        self.clip_model.float()

        nuclei_nums = {'10', '45', '80', '115', '150', '185'}
        self.coop_num = coop.CustomCLIP_num(nuclei_nums, self.clip_model)
        for name, param in self.coop_num.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # self.coop_tc = coop.CustomCLIP_tc(self.CLASSES, self.clip_model)
        # for name, param in self.coop_tc.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)
        # self.coop_nc = coop.CustomCLIP_nc(self.CLASSES, self.clip_model)
        # for name, param in self.coop_nc.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)

        ###clip_adapter###
        # nuclei_nums = {'10', '45', '80', '115', '150', '185'}
        # self.clip_adapter_num = clip_adapter.CustomCLIP(nuclei_nums, self.clip_adapter_model)
        # for name, param in self.clip_adapter_num.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)

        self.clip_adapter_model_tc = clip_adapter.load_clip_to_cpu_tc(pretrained_model)
        self.clip_adapter_model_tc.float()
        self.clip_adapter_tc = clip_adapter.CustomCLIP_tc(self.CLASSES, self.clip_adapter_model_tc)
        for name, param in self.clip_adapter_tc.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
        self.clip_adapter_model_tc.to(gpu_ids[0])

        self.clip_adapter_model_nc = clip_adapter.load_clip_to_cpu_nc(pretrained_model)
        self.clip_adapter_model_nc.float()
        self.clip_adapter_nc = clip_adapter.CustomCLIP_nc(self.CLASSES, self.clip_adapter_model_nc)
        for name, param in self.clip_adapter_nc.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
        self.clip_adapter_model_nc.to(gpu_ids[0])

        self.clip_adapter_model_sc = clip_adapter.load_clip_to_cpu_sc(pretrained_model)
        self.clip_adapter_model_sc.float()
        self.clip_adapter_sc = clip_adapter.CustomCLIP_sc(self.CLASSES, self.clip_adapter_model_nc)
        for name, param in self.clip_adapter_sc.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
        self.clip_adapter_model_sc.to(gpu_ids[0])

    def create_decoder_branch(self, out_ch=2, ksize=5):
        pad = ksize // 2
        module_list = [
            nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
            DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
        ]
        u3 = nn.Sequential(*module_list)

        module_list = [
            nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
            DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
        ]
        u2 = nn.Sequential(*module_list)

        module_list = [
            nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
        ]
        u1 = nn.Sequential(*module_list)

        module_list = [
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
        ]
        u0 = nn.Sequential(*module_list)

        decoder = nn.Sequential(OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)]))
        return decoder

    def decoder_forward(self, x, decoder):
        u3 = self.upsample2x(x[-1]) + x[-2]
        u3 = decoder[0](u3)

        u2 = self.upsample2x(u3) + x[-3]
        u2 = decoder[1](u2)

        u1 = self.upsample2x(u2) + x[-4]
        u1 = decoder[2](u1)

        u0 = decoder[3](u1)

        return u0

    def calculate(self, x):
        if self.training:
            d0, d1, d2, d3, d4 = self.backbone(x)
        else:
            d0, d1, d2, d3, d4 = self.backbone(x)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]
        # sem_logit_cls = self.decoder_forward(d, self.decoder['tp']) # 类别分支 torch.Size([8, 3, 256, 256])
        tc_logit = self.decoder_forward(d, self.decoder['tc'])  # 类别clip分支 torch.Size([8, 1024, 256, 256])
        hv_logit = self.decoder_forward(d, self.decoder['hv'])  # 水平和垂直映射分支 torch.Size([8, 2, 256, 256])
        # fore_logit = self.decoder_forward(d, self.decoder['np']) # 核像素预测分支 torch.Size([8, 2, 256, 256])
        nc_logit = self.decoder_forward(d, self.decoder['nc'])  # 核像素clip预测分支 torch.Size([8, 1024, 256, 256])
        sc_logit = self.decoder_forward(d, self.decoder['sc']) # 类别分支 torch.Size([8, 3, 256, 256])

        ###clip预处理###
        # number_clip
        # logits_per_image, logits_per_text = self.clip_process_num(x, d4)
        logits_per_image = None
        logits_per_text = None

        # inst_color_gt = label['inst_color_gt'].squeeze(1)
        # inst_num_gt = self.get_inst_num(inst_color_gt)
        # classfication_clip
        nc_logit = self.clip_process_nc(x, nc_logit)
        tc_logit = self.clip_process_tc(x, tc_logit)
        sem_logit_cls = self.clip_process_sc(x, sc_logit)

        # sem_logit_cls = tc_logit
        # 8-to-2
        fore_logit = self.nc_conv(nc_logit)
        sem_logit = self.tc_conv(tc_logit)

        if not self.training:
            return sem_logit, sem_logit_cls, hv_logit, fore_logit

        return sem_logit, sem_logit_cls, hv_logit, fore_logit, logits_per_image, logits_per_text

    def get_bboxes(self, inst_color_gt):
        img_array = inst_color_gt.cpu().numpy()
        unique_colors, color_counts = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0, return_counts=True)
        empty_matrix = torch.zeros((img_array.shape[0], img_array.shape[1]), dtype=torch.float32)

        color_dict = {}
        for color in unique_colors:
            color_indices = np.where(np.all(img_array == color, axis=-1))
            empty_matrix[color_indices] = color_dict.setdefault(tuple(color), len(color_dict))

        inst_ids = [inst_id for inst_id in color_dict.keys() if inst_id != (0,)]

        bboxes = []
        for inst_id in inst_ids:
            if inst_id == (0, 0, 0):  # 跳过背景
                continue
            inst_map = empty_matrix == color_dict[inst_id]

            binary = inst_map.int()[:, :].cpu().numpy()
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x0 = x
                x1 = x + w
                y0 = y
                y1 = y + h
                bboxes.append([x0, x1, y0, y1])

        return np.array(bboxes)

    def get_inst_bboxes(self, inst_color_gt):
        inst_bboxes_gt = []
        # labels_gt = []
        for i in range(0, inst_color_gt.shape[0]):
            inst_bboxes_gt_i = torch.tensor(self.get_bboxes(inst_color_gt[i, :, :])[:, [0, 2, 1, 3]])
            # labels_gt_i = [torch.zeros(bboxe_gt.shape[0],dtype=torch.long) for bboxe_gt in bboxes_gt_i]
            inst_bboxes_gt.append(inst_bboxes_gt_i)
            # labels_gt.append(labels_gt_i)
        return inst_bboxes_gt

    def get_num(self, inst_color_gt):
        img_array = inst_color_gt.cpu().numpy()
        unique_colors, color_counts = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0, return_counts=True)
        empty_matrix = torch.zeros((img_array.shape[0], img_array.shape[1]), dtype=torch.float32)

        color_dict = {}
        for color in unique_colors:
            color_indices = np.where(np.logical_and.reduce(img_array == color, axis=-1))
            empty_matrix[color_indices] = color_dict.setdefault(tuple(color), len(color_dict))

        inst_ids = [inst_id for inst_id in color_dict.keys() if inst_id != (0,)]

        inst_num = 0
        for inst_id in inst_ids:
            if inst_id == (0, 0, 0):  # 跳过背景
                continue
            inst_map = empty_matrix == color_dict[inst_id]
            binary = inst_map.int()[:, :].cpu().numpy()
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            inst_num += len(contours)
        return inst_num

    def get_inst_num(self, inst_color_gt):
        inst_num_gt = []
        for i in range(0, inst_color_gt.shape[0]):
            inst_num_gt.append(self.get_num(inst_color_gt[i, :, :]))
        return inst_num_gt

    def get_type_class(self, type_num_gt):
        # class_values = {
        #     1: "other",
        #     2: "inflammatory",
        #     3: "healthy epithelial",
        #     4: "dysplastic or malignant epithelial",
        #     5: "fibroblast",
        #     6: "muscle",
        #     7: "endothelial"
        # }

        # type_class_gt = [
        #     [class_values[value.item()] for value in type_num_gt[i, :, :][type_num_gt[i, :, :] != 0].unique()]
        #     for i in range(type_num_gt.shape[0])
        # ]

        type_class_gt = []
        for i in range(type_num_gt.shape[0]):
            unique_values = type_num_gt[i, :, :][type_num_gt[i, :, :] != 0].unique()
            type_class_gt.append(unique_values)

        return type_class_gt

    def convert_to_one_hot(self, type_num_gt):
        indexes = [torch.unique(type_num_gt[i][type_num_gt[i] != 0]) for i in range(type_num_gt.shape[0])]
        one_hot = torch.zeros((type_num_gt.shape[0], self.num_classes), dtype=torch.long, device=type_num_gt.device)
        one_hot[:, 0] = 1
        for i, idx in enumerate(indexes):
            one_hot[i, idx] = 1
        return one_hot

    def clip_process_num(self, image, image_features):
        ###prompt_align###
        # logits_per_image_coop, logits_per_text_coop = self.coop_num(image_features, image)

        # crowd_count = ['2', '8', '16', '27', '41', '57']
        # text_list = [f"There are {c} nucleus in the picture" for c in crowd_count]
        # tokenized_text_list = [clip.tokenize(text) for text in text_list]
        # text = torch.cat(tokenized_text_list * image.shape[0]).to(image.device)

        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text)
        #     text_features = text_features / text_features.norm(dim=1, keepdim=True)
        #     text_features = text_features.float()

        # image_features = torch.stack(image_features, dim=1)
        # image_features = image_features.view(image.shape[0] * 6, 1024)
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # logits_per_image_clip = self.logit_scale_num.exp() * image_features @ text_features.t()

        # return logits_per_image_coop, logits_per_text_coop, logits_per_image_clip
        ###coop###
        # logits_per_image, logits_per_text = self.coop_num(image_features, image)

        ###clip_adapter###
        # logits_per_image, logits_per_text = self.clip_adapter_num(image_features)

        ###clip###
        # crowd_count = ['2', '8', '16', '27', '41', '57']
        # text_list = [f"There are {c} nucleus in the picture" for c in crowd_count]
        # tokenized_text_list = [clip.tokenize(text) for text in text_list]
        # text = torch.cat(tokenized_text_list * image.shape[0]).to(image.device)
        #
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text)
        #     text_features = text_features / text_features.norm(dim=1, keepdim=True)
        #     text_features = text_features.float()
        #
        # image_features = torch.stack(image_features, dim=1)
        # image_features = image_features.view(image.shape[0] * 6, 1024)
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # logits_per_image = self.logit_scale_num.exp() * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        ###plip###
        # crowd_count = ['2', '8', '16', '27', '41', '57']
        # text_list = [f"There are {c} nucleus in the picture" for c in crowd_count]
        
        # plip = PLIP('/data/rubyyao/PycharmProjects/nuclei_ins_seg/pretrained/vinid/plip')
        # with torch.no_grad():
        #     text_features = plip.encode_text(text_list, batch_size=1)
        # np.save('/data/rubyyao/PycharmProjects/nuclei_ins_seg/10_tiseg+lseg_img_encoder_for_consep_with_class_metric_5_2_00/text_features.npy', text_features)
        text_features = np.load('/root/autodl-tmp/rubyyao/PycharmProjects/nuclei_ins_seg/code/10_tiseg+lseg_img_encoder_for_consep_with_class_metric_6_3/text_features.npy')
        concatenated_text_features = np.concatenate((text_features, text_features, text_features, text_features),
                                                    axis=0)
        text_features = torch.from_numpy(concatenated_text_features).to(image.device)
        text_features /= torch.norm(text_features, p=2, dim=-1, keepdim=True)
        
        image_features = torch.stack(image_features, dim=1)
        image_features = image_features.view(image.shape[0] * 6, 512)
        image_features = image_features / torch.norm(image_features, p=2, dim=-1, keepdim=True)
        logits_per_image = self.logit_scale_num.exp() * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ###cal_similarity_acc###
        # y_pred = logits_per_image.detach().cpu().numpy()
        # count = 0
        # for idx, row in enumerate(y_pred):
        #     if idx in np.where(row == np.max(row))[0]:
        #         count += 1
        # accuracy = count / len(y_pred)
        #
        # print("准确率:", accuracy)
        ####

        return logits_per_image, logits_per_text


    def clip_process_tc(self, image, image_features):
        ###coop###
        # out = self.coop_tc(image_features)

        ###clip_adapter###
        out = self.clip_adapter_tc(image_features)

        ###Lseg###
        # text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.CLASSES]).to(image.device)
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text)

        # imshape = image_features.shape
        # image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)

        # # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits_per_image = self.logit_scale_tc.exp() * image_features @ text_features.t()

        # out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        ###cal_similarity_acc###
        # y_pred = logits_per_image.detach().cpu().numpy()
        # count = 0
        # for idx, row in enumerate(y_pred):
        #     if idx in np.where(row == np.max(row))[0]:
        #         count += 1
        # accuracy = count / len(y_pred)
        #
        # print("准确率:", accuracy)
        ####

        return out

    def clip_process_nc(self, image, image_features):
        ###CoOP###
        # out = self.coop_nc(image_features)

        ###clip_adapter###
        out = self.clip_adapter_nc(image_features)

        ##Lseg###
        # text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.CLASSES]).to(image.device)
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text)
        #
        # imshape = image_features.shape
        # image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        #
        # # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #
        # logits_per_image = self.logit_scale_nc.exp() * image_features.half() @ text_features.t()
        #
        # out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        ###cal_similarity_acc###
        # y_pred = logits_per_image.detach().cpu().numpy()
        # count = 0
        # for idx, row in enumerate(y_pred):
        #     if idx in np.where(row == np.max(row))[0]:
        #         count += 1
        # accuracy = count / len(y_pred)
        #
        # print("准确率:", accuracy)
        ####

        return out

    def clip_process_sc(self, image, image_features):
        ###CoOP###
        # out = self.coop_nc(image_features)

        ###clip_adapter###
        out = self.clip_adapter_sc(image_features)

        ##Lseg###
        # text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.CLASSES]).to(image.device)
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text)
        #
        # imshape = image_features.shape
        # image_features = image_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        #
        # # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #
        # logits_per_image = self.logit_scale_nc.exp() * image_features.half() @ text_features.t()
        #
        # out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        ###cal_similarity_acc###
        # y_pred = logits_per_image.detach().cpu().numpy()
        # count = 0
        # for idx, row in enumerate(y_pred):
        #     if idx in np.where(row == np.max(row))[0]:
        #         count += 1
        # accuracy = count / len(y_pred)
        #
        # print("准确率:", accuracy)
        ####

        return out
    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_logit, sem_logit_cls, hv_logit, fore_logit, logits_per_image, logits_per_text = self.calculate(data['img'])
            # sem_logit, hv_logit, fore_logit, logits_per_image_coop, logits_per_text_coop, logits_per_image_clip = self.calculate(data['img'])
            # print("The code is 10_tiseg+lseg_img_encoder_for_consep_with_class_metric_only_instance_level_7_15_01")

            # imshow_img_mask/num_gt#
            # import matplotlib.pyplot as plt
            # import matplotlib
            # from PIL import Image, ImageDraw
            # import os
            #
            # image_np = label['inst_gt'][0, :, :, :].cpu().numpy()
            # plt.imshow(image_np)
            # plt.show()
            #
            # image_np = label['inst_color_gt'][0, :, :, :].cpu().numpy()
            # plt.imshow(image_np)
            # plt.show()
            #
            # image_np = data['img'][0, :, :, :].cpu().numpy()
            # image_np = np.transpose(image_np, (1, 2, 0))
            # plt.imshow(image_np)
            # plt.show()
            #
            # img_path = "/data/rubyyao/PycharmProjects/nuclei_ins_seg/a.jpg"
            # matplotlib.image.imsave(img_path, image_np)
            # green = (0, 255, 0)
            # image = Image.open(img_path)
            # os.remove(img_path)
            # draw = ImageDraw.Draw(image)
            # inst_bboxes_gt = self.get_inst_bboxes(inst_color_gt)
            # for i in range(0, inst_bboxes_gt[0].shape[0]):
            #     x0 = inst_bboxes_gt[0][i, 0]
            #     y0 = inst_bboxes_gt[0][i, 1]
            #     x1 = inst_bboxes_gt[0][i, 2]
            #     y1 = inst_bboxes_gt[0][i, 3]
            #     draw.line([(x0, y0), (x0, y1)], fill=green, width=3)
            #     draw.line([(x0, y1), (x1, y1)], fill=green, width=3)
            #     draw.line([(x1, y1), (x1, y0)], fill=green, width=3)
            #     draw.line([(x1, y0), (x0, y0)], fill=green, width=3)
            # plt.imshow(image)
            # plt.show()
            ####

            assert label is not None
            sem_gt = label['sem_gt']
            type_gt = label['type_gt']
            hv_gt = label['hv_gt']
            fore_gt = sem_gt.clone()
            fore_gt = (sem_gt > 0).long()

            loss = dict()

            sem_gt = sem_gt.squeeze(1)
            fore_gt = fore_gt.squeeze(1)
            type_gt = type_gt.squeeze(1)

            # TODO: Conside to remove some edge loss value.

            # mask branch loss calculation(tp)
            sem_loss = self._sem_loss(sem_logit, sem_gt)
            loss.update(sem_loss)

            sem_cls_loss = self._sem_cls_loss(sem_logit_cls, type_gt)
            loss.update(sem_cls_loss)

            # direction branch loss calculation(hv)
            hv_loss = self._hv_loss(hv_logit, hv_gt, fore_gt)
            loss.update(hv_loss)
            # point branch loss calculation(np)
            fore_loss = self._fore_loss(fore_logit, fore_gt)
            loss.update(fore_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, fore_logit, sem_gt, fore_gt)
            loss.update(training_metric_dict)

            # clip_num loss calculation
            # clip_num_loss = self.clip_num_loss(logits_per_image, logits_per_text, data['img'].shape[0])
            # loss.update(clip_num_loss)

            # prompt_align loss
            # prompt_align_loss = self.prompt_align_loss(logits_per_image_coop, logits_per_image_clip.detach())
            # loss.update(prompt_align_loss)
            ####

            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            sem_logit, sem_logit_cls, hv_logit, fore_logit = self.inference(data['img'], metas[0], True)



            ################### ignore please !!! ################
            ################### only used for classification metric evaluation ################
            pred_dict = OrderedDict()
            pred_dict['tp'] = sem_logit_cls
            pred_dict['np'] = fore_logit
            pred_dict['hv'] = hv_logit
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            classification_pred = torch.cat(list(pred_dict.values()), -1).cpu().numpy()
            ################### only used for classification metric evaluation ################



            type_pred = sem_logit.argmax(dim=1)
            # Extract inside class
            type_pred = type_pred.cpu().numpy()[0]
            sem_pred = np.zeros_like(type_pred)
            sem_pred[type_pred != 0] = 1
            # NHW -> HWN
            hv_pred = hv_logit.permute(0, 2, 3, 1).cpu().numpy()[0]
            fore_logit = fore_logit.cpu().numpy()[0][1]
            # unravel batch dim
            inst_pred = self.hover_post_proc(fore_logit, hv_pred, scale_factor=self.test_cfg.get('scale_factor', 1))
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred, 'classification_pred': classification_pred})
            return ret_list

    def hover_post_proc(self, fore_map, hv_map, fx=1, scale_factor=1):
        raw_h, raw_w = hv_map.shape[:2]

        fore_map = cv2.resize(fore_map, (0, 0), fx=scale_factor, fy=scale_factor)
        hv_map = cv2.resize(hv_map, (0, 0), fx=scale_factor, fy=scale_factor)

        blb_raw = fore_map
        h_dir_raw = hv_map[:, :, 0]
        v_dir_raw = hv_map[:, :, 1]

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)

        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        ksize = int((20 * fx) + 1)
        obj_size = math.ceil(10 * (fx ** 2))
        # Get resolution specific filters etc.

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            ))
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            ))

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        # * nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=obj_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        proced_pred = cv2.resize(proced_pred, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

        return proced_pred

    def inference(self, img, meta, rescale):
        """Inference with split/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['split', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        sem_logit_list = []
        hv_logit_list = []
        fore_logit_list = []
        sem_logit_cls_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    sem_logit, sem_logit_cls, hv_logit, fore_logit = self.split_inference(img, meta, rescale)
                else:
                    sem_logit, hv_logit, fore_logit = self.whole_inference(img, meta, rescale)

                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                sem_logit_cls = self.reverse_tta_transform(sem_logit_cls, rotate_degree, flip_direction)
                hv_logit = self.reverse_tta_transform(hv_logit, rotate_degree, flip_direction)
                fore_logit = self.reverse_tta_transform(fore_logit, rotate_degree, flip_direction)

                sem_logit_cls = F.softmax(sem_logit_cls, dim=1)
                sem_logit = F.softmax(sem_logit, dim=1)
                fore_logit = F.softmax(fore_logit, dim=1)

                sem_logit_cls_list.append(sem_logit_cls)
                sem_logit_list.append(sem_logit)
                hv_logit_list.append(hv_logit)
                fore_logit_list.append(fore_logit)

        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        sem_logit_cls = sum(sem_logit_cls_list) / len(sem_logit_cls_list)
        hv_logit = hv_logit_list[0]
        fore_logit = sum(fore_logit_list) / len(fore_logit_list)

        if rescale:
            sem_logit_cls = resize(sem_logit_cls, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            hv_logit = resize(hv_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            fore_logit = resize(fore_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return sem_logit,sem_logit_cls, hv_logit, fore_logit

    def split_inference(self, img, meta, rescale):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = self.test_cfg.overlap_size[0]

        B, C, H, W = img.shape

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (window_size - overlap_size)
        else:
            pad_h = window_size - H

        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (window_size - overlap_size)
        else:
            pad_w = window_size - W

        H1, W1 = pad_h + H, pad_w + W
        img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
        img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

        sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        sem_logit_cls = torch.zeros((B, 8, H1, W1), dtype=img.dtype, device=img.device)
        hv_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)
        fore_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                sem_patch, sem_logit_cls_patch, hv_patch, fore_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                ind2_s - j:ind2_e - j]
                sem_logit_cls[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_logit_cls_patch[:, :, ind1_s - i:ind1_e - i,
                                                                ind2_s - j:ind2_e - j]
                hv_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = hv_patch[:, :, ind1_s - i:ind1_e - i,
                                                               ind2_s - j:ind2_e - j]
                fore_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = fore_patch[:, :, ind1_s - i:ind1_e - i,
                                                                 ind2_s - j:ind2_e - j]

        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        hv_logit = hv_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        fore_logit = fore_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]

        return sem_logit, sem_logit_cls, hv_logit, fore_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        sem_logit, hv_logit, fore_logit, clip_logit = self.calculate(img)

        return sem_logit, hv_logit, fore_logit

    def _sem_loss(self, sem_logit, sem_gt, weight_map=None):
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt)
        if weight_map is not None:
            sem_ce_loss *= weight_map[:, 0]
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss

    def _sem_cls_loss(self, sem_logit, sem_gt, weight_map=None):
        sem_cls_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=8)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt)
        if weight_map is not None:
            sem_ce_loss *= weight_map[:, 0]
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        sem_cls_loss['sem_cls_ce_loss'] = alpha * sem_ce_loss
        sem_cls_loss['sem_cls_dice_loss'] = beta * sem_dice_loss

        return sem_cls_loss
    def _hv_loss(self, hv_logit, hv_gt, fore_gt):
        hv_loss = {}
        hv_mse_loss_calculator = nn.MSELoss()
        hv_msge_loss_calculator = GradientMSELoss()
        hv_mse_loss = hv_mse_loss_calculator(hv_logit, hv_gt)
        hv_msge_loss = hv_msge_loss_calculator(hv_logit, hv_gt, fore_gt)
        # loss weight
        alpha = 1
        beta = 1
        hv_loss['hv_mse_loss'] = alpha * hv_mse_loss
        hv_loss['hv_msge_loss'] = beta * hv_msge_loss

        return hv_loss

    def _fore_loss(self, fore_logit, fore_gt):
        fore_loss = {}
        fore_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        fore_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=2)
        fore_ce_loss = fore_ce_loss_calculator(fore_logit, fore_gt)
        fore_ce_loss = torch.mean(fore_ce_loss)
        fore_dice_loss = fore_dice_loss_calculator(fore_logit, fore_gt)
        # loss weight
        alpha = 3 * 1
        beta = 3 * 1
        fore_loss['fore_ce_loss'] = alpha * fore_ce_loss
        fore_loss['fore_dice_loss'] = beta * fore_dice_loss

        return fore_loss

    def _training_metric(self, sem_logit, fore_logit, sem_gt, fore_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_sem_logit = sem_logit.clone().detach()
        clean_sem_gt = sem_gt.clone().detach()
        clean_fore_logit = fore_logit.clone().detach()
        clean_fore_gt = fore_gt.clone().detach()

        wrap_dict['sem_mdice'] = mdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['fore_mdice'] = mdice(clean_fore_logit, clean_fore_gt, 2)

        wrap_dict['sem_tdice'] = tdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['fore_tdice'] = tdice(clean_fore_logit, clean_fore_gt, 2)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # sem_pred = torch.argmax(sem_logit, dim=1).cpu().numpy().astype(np.uint8)
        # sem_pred[sem_pred == (self.num_classes - 1)] = 0
        # sem_target = sem_gt.cpu().numpy().astype(np.uint8)
        # sem_target[sem_target == (self.num_classes - 1)] = 0

        # N = sem_pred.shape[0]
        # wrap_dict['aji'] = 0.
        # for i in range(N):
        #     aji_single_image = aggregated_jaccard_index(sem_pred[i], sem_target[i])
        #     wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # # distributed environment requires cuda tensor
        # wrap_dict['aji'] /= N
        # wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def clip_num_loss(self, logits_per_image, logits_per_text, batch_size):
        clip_num_loss = {}
        # cross entropy loss
        loss_text = F.cross_entropy(logits_per_text, torch.arange(batch_size * 6, device=logits_per_text.device))
        loss_image = F.cross_entropy(logits_per_image, torch.arange(batch_size * 6, device=logits_per_image.device))
        mean_loss = (loss_text + loss_image) / 2

        # CoOP_loss
        # clip_loss = {}
        # loss_func = nn.MultiLabelSoftMarginLoss()
        # loss = loss_func(logits_per_image, clip_gt)
        # clip_loss['clip_loss'] = loss

        ###cal_similarity_acc###
        # y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        # y_pred = np.argmax(logits_per_image.detach().cpu().numpy(), axis=1)
        # accuracy = np.mean(y_true == y_pred)
        #
        # print("准确率：", accuracy)
        ####

        clip_num_loss['clip_num_loss'] = mean_loss
        return clip_num_loss

    def prompt_align_loss(self, stu_logits, tea_logits):
        prompt_align_loss = {}
        self.T = 1.0
        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()
        prompt_align_loss['prompt_align_loss'] = kl_loss
        return prompt_align_loss
