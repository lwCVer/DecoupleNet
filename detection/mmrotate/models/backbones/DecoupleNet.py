import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import copy
import os
import antialiased_cnns
import torch.nn.functional as F
from ..builder import ROTATED_BACKBONES


try:
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


# FID is feature integration downsampling
class FID(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.Gconv = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.pii = PII(dim*2, 8)
        self.conv_D = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, groups=dim*2)
        self.act = nn.GELU()
        self.batch_norm_c = nn.BatchNorm2d(dim*2)
        self.max_m1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.max_m2 = antialiased_cnns.BlurPool(dim*2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(dim*2)
        self.fusion = nn.Conv2d(dim*4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]

        # Gconv + PII
        x = self.Gconv(x)  # h = [B, 2C, H, W]
        x = self.pii(x)

        # MaxD + anti-aliased
        max = self.max_m1(x)  # m = [B, 2C, H/2, W/2]
        max = self.max_m2(max)  # m = [B, 2C, H/2, W/2]
        max = self.batch_norm_m(max)

        # ConvD
        conv = self.conv_D(x)  # h = [B, 2C, H/2, W/2]
        conv = self.act(conv)
        conv = self.batch_norm_c(conv)  # h = [B, 2C, H/2, W/2]

        # Concat
        x = torch.cat([conv, max], dim=1)  # x = [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]  -->  [B, 2C, H/2, W/2]

        return x


class PII(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = int((dim / 2) - self.dim_conv)
        self.conv = nn.Conv2d(self.dim_conv*2, self.dim_conv*2, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:

        x1, x2, x3, x4 = torch.split(x, [self.dim_conv, self.dim_untouched, self.dim_conv, self.dim_untouched], dim=1)
        x = torch.cat((x1, x3), 1)
        x1 = self.conv(x)
        x = torch.cat((x1, x2, x4), 1)

        return x


# MRLA is medium-range lightweight Attention
class MRLA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super(MRLA, self).__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        channels12 = int(channel / 2)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(channel, channels12, 1, 1, bias=False),
            norm_layer(channels12),
            nn.GELU(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(channels12, channels12, 3, 1, 1, groups=channels12, bias=False),
            norm_layer(channels12),
            nn.GELU(),
        )
        self.init = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.H_att = nn.Conv2d(channel, channel, (att_kernel, 1), 1, (att_padding, 0), groups=channel, bias=False)
        self.V_att = nn.Conv2d(channel, channel, (1, att_kernel), 1, (0, att_padding), groups=channel, bias=False)
        self.batchnorm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.init(F.avg_pool2d(x, kernel_size=2, stride=2))
        x_h = self.H_att(x_tem)
        x_w = self.V_att(x_tem)
        mrla = self.batchnorm(x_h + x_w)

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out[:, :self.channel, :, :] * F.interpolate(self.gate_fn(mrla),
                                                          size=(out.shape[-2], out.shape[-1]),
                                                          mode='nearest')
        return out


# GA is long range Attention
class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


# MBFD is multi-branch feature decoupling module
class MBFD(nn.Module):

    def __init__(self, dim, stage, att_kernel, norm_layer):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.dim_learn = dim // 4
        self.dim_untouched = dim - self.dim_learn - self.dim_learn
        self.Conv = nn.Conv2d(self.dim_learn, self.dim_learn, 3, 1, 1, bias=False)
        self.MRLA = MRLA(self.dim_learn, att_kernel, norm_layer)  # MRLA is medium range Attention
        if stage > 2:
            self.GA = GA(self.dim_untouched)      # GA is long range Attention
            self.norm = norm_layer(self.dim_untouched)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2, x3 = torch.split(x, [self.dim_learn, self.dim_learn, self.dim_untouched], dim=1)
        x1 = self.Conv(x1)
        x2 = self.MRLA(x2)
        if self.stage > 2:
            x3 = self.norm(x3 + self.GA(x3))
        x = torch.cat((x1, x2, x3), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 stage,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.MBFD = MBFD(
            dim,
            stage,
            att_kernel,
            norm_layer
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.MBFD(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.MBFD(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 stage,
                 depth,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                stage=stage,
                att_kernel=att_kernel,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


@ROTATED_BACKBONES.register_module()
class DecoupleNet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=32,
                 depths=(1, 6, 6, 2),
                 att_kernel=(9, 9, 9, 9),
                 mlp_ratio=2.,
                 patch_size=4,
                 patch_stride=4,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.GELU,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        norm_layer = norm_layer
        act_layer = act_layer

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.att_kernel = att_kernel

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               stage=i_stage,
                               depth=depths[i_stage],
                               att_kernel=att_kernel[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    FID(dim=int(embed_dim * 2 ** i_stage))
                )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) \
                if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs


def DecoupleNet_D0_1662_e32_k9_drop0(num_classes: int = 1000, **kwargs):
    model = DecoupleNet(in_chans=3,
                      num_classes=num_classes,
                      embed_dim=32,
                      depths=(1, 6, 6, 2),
                      att_kernel=(9, 9, 9, 9),
                      drop_path_rate=0.0,
                      **kwargs)

    return model


def DecoupleNet_D1_1662_e48_k9(num_classes: int = 1000, **kwargs):
    model = DecoupleNet(in_chans=3,
                      num_classes=num_classes,
                      embed_dim=48,
                      depths=(1, 6, 6, 2),
                      att_kernel=(9, 9, 9, 9),
                      drop_path_rate=0.1,
                      **kwargs)

    return model


def DecoupleNet_D2_1662_e64_k9_drop01(num_classes: int = 1000, **kwargs):
    model = DecoupleNet(in_chans=3,
                      num_classes=num_classes,
                      embed_dim=64,
                      depths=(1, 6, 6, 2),
                      att_kernel=(9, 9, 9, 9),
                      drop_path_rate=0.1,
                      **kwargs)

    return model
