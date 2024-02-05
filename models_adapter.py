# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from typing import Tuple
from collections import OrderedDict
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_L14_PATH,
    DWCONV3D_DISABLE_CUDNN,
    )

class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.).half()
        nn.init.constant_(self.conv.bias, 0.).half()
        nn.init.constant_(self.fc1.bias, 0.).half()
        nn.init.constant_(self.fc2.bias, 0.).half()

    def forward(self, x, T):
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += x
        return x_id


class expand_tubevit(nn.Module):
    def __init__(self, scale, width):
        super(expand_tubevit, self).__init__()
        self.spatial_start_point = [45, 48, 87, 90]
        self.patch_size = [3, 5, 7, 9]
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

    def get_patch_index(self, x, spatial_point, patch_size):
        if patch_size == 3:
            sp = spatial_point
        elif patch_size == 5:
            sp = spatial_point - 15
        elif patch_size == 7:
            sp = spatial_point - (15 * 2)
        elif patch_size == 9:
            sp = spatial_point - (15 * 3)

        patch_gap = (patch_size + 1) // 2
        additional_pathes = [sp, sp+patch_gap, sp+(patch_gap*2),
                                sp+(14 * patch_gap), sp+(14*patch_gap)+((patch_gap*2)),
                                sp+(14 * patch_gap*2), sp+(14 * patch_gap*2)+patch_gap, sp+(14 * patch_gap*2)+(patch_gap*2)]
        center_patch = [(14 * i) + sp + j + 1 for j in range(patch_size) for i in range(0, patch_size)] # j: 각 patch를 grid하게 탐색, i: 각 patch의 시작 point를 찾음
        selected_patch = additional_pathes + center_patch
        selected_patch.sort()
        selected_patch = torch.index_select(x.cpu(), 0, torch.tensor(selected_patch)).cuda()
        return selected_patch
    
    def forward(self, x):
        expand_tube = torch.empty(0, x.size(1), x.size(2), x.size(3)).cuda()

        for batch in range(x.size(0)):
            tube = torch.empty(0, x.size(2), x.size(3)).cuda()
            for frame_idx in range(x.size(1)):
                # 현재 프레임 선택
                distribute_frame = frame_idx % 4
                if distribute_frame == 0:
                    for spatial_point in self.spatial_start_point:
                        st_tube = torch.empty(0,768).cuda()
                        for i, idx in enumerate(range(frame_idx, frame_idx+4 , 1)):
                            current_frame = x[batch, idx, :, :]
                            selected_frame = self.get_patch_index(current_frame, spatial_point, self.patch_size[i])
                            st_tube = torch.cat([st_tube, selected_frame], dim=0)
                        st_tube = st_tube.unsqueeze(0)
                        tube = torch.cat([tube, st_tube], dim=0)
                else:
                    pass
            tube = tube.unsqueeze(0)
            expand_tube = torch.cat([expand_tube, tube], dim=0)
        return expand_tube

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        adapter_class = functools.partial(
            Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=adapter_kernel_size,
        )
        self.adapter_pre_attn = \
            adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = \
            adapter_class() if adapter_pre_mlp else None

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        H = self.attn.num_heads

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        return out
        

    def forward(self,
                x: torch.Tensor,
                num_frames: int
                ) -> torch.Tensor:
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        x = x + self.attention(self.ln_1(x))
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, num_frames)
        return x


class expand_st_adapter(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 num_classes: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 classifier: str,
                 testing=None
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        
        self.spatial_positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 , width
            )
        )

        self.temporal_positional_embedding = nn.Parameter(
            scale * torch.randn(
                8, ((input_resolution // patch_size) ** 2) * width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
            adapter_width, adapter_layers, adapter_kernel_size,
            adapter_pre_attn, adapter_pre_mlp)
        self.expand_tubevit = expand_tubevit(scale, width)

        for n, p in self.named_parameters():
          if 'adapter' not in n:
            p.requires_grad_(True)
            p.data = p.data.half()
        
        self.dropout = nn.Dropout(0.5)

        self.classifier = classifier
        if self.classifier == 'mean' or self.classifier == 'difference':
            self.expand_fc = nn.Linear(width, num_classes)
            self.ln_expand_post = LayerNorm(width)
        elif self.classifier == 'span2':
            self.expand_fc = nn.Linear(width * 2, num_classes)
            self.ln_expand_post = LayerNorm(width * 2)
        nn.init.normal_(self.expand_fc.weight, std=0.02)
        nn.init.constant_(self.expand_fc.bias, 0.)

        self.testing = testing

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)
        x = x + self.spatial_positional_embedding.to(x.dtype)
        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1], x.size(-1))
    
        # Temporal positional embedding
        S_patch, embedded_patch = x.size(2), x.size(3)
        x = x.view(x.size(0), x.size(1), -1)
        x = x + self.temporal_positional_embedding.to(x.dtype)
        x = x.contiguous().view(B, T, S_patch, embedded_patch)

        # Expand tube vit
        x = self.expand_tubevit(x)
        x = x.flatten(0, 1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        
        x = self.ln_pre(x)
        x = self.transformer(x, T)

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        if self.testing is not None:
            new_tensor = torch.cat([x[:, i, 0, :] for i in range(x.size(1))], dim=0)
        # St_adapter의 방식으로 모든 것을 평균내버리는 방식
        if self.classifier == 'mean':
            x = x[:, :, 0, :].mean(dim=1)
        elif self.classifier == 'span2':
            x = torch.stack((x[:, 0:T//2, 0, :].mean(dim=1), x[:, T//2:, 0, :].mean(dim=1))).view(B, x.size()[3] * 2)
        elif self.classifier == 'difference':
            x = x[:, 0:T//2, 0, :].mean(dim=1) - x[:, T//2:, 0, :].mean(dim=1)
        x = self.ln_expand_post(x)
        x = self.dropout(x)
        x = self.expand_fc(x)

        if self.testing is not None:
            return x, new_tensor
        else:
            return x


class st_adapter_VIT(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 num_classes: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 testing=None
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
            adapter_width, adapter_layers, adapter_kernel_size,
            adapter_pre_attn, adapter_pre_mlp)

        self.ln_post = LayerNorm(width)

        for n, p in self.named_parameters():
          if 'adapter' not in n:
            p.requires_grad_(True)
            p.data = p.data.half()
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(width, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)

        self.testing = testing
    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D

        x = self.transformer(x, T)

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        if self.testing is not None:
            new_tensor = torch.cat([x[:, i, 0, :] for i in range(x.size(1))], dim=0)
        x = x[:, :, 0, :].mean(dim=1)
        x = self.ln_post(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.testing is not None:
            return x, new_tensor
        else:
            return x


def clip_vit_base_patch16_adapter24x384(model_name, classifier=None, **kwargs):
    if model_name == 'st_adapter':
        model = st_adapter_VIT(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    
    elif model_name == 'expand_st_adapter':
        model = expand_st_adapter(
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            adapter_width=384,
            adapter_layers=12,
            adapter_kernel_size=(3, 1, 1),
            adapter_pre_attn=True,
            adapter_pre_mlp=True,
            classifier=classifier,
            **kwargs,
        )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    checkpoint = torch.jit.load('/hahmwj/ViT-B-16.pt', map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model

def clip_vit_base_patch16_adapter12x384(model_name, classifier=None, **kwargs):
    if model_name == 'st_adapter':
        model = st_adapter_VIT(
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            adapter_width=384,
            adapter_layers=12,
            adapter_kernel_size=(3, 1, 1),
            adapter_pre_attn=False,
            adapter_pre_mlp=True,
            **kwargs,
        )
    elif model_name == 'expand_st_adapter':
        model = expand_st_adapter(
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            adapter_width=384,
            adapter_layers=12,
            adapter_kernel_size=(3, 1, 1),
            adapter_pre_attn=False,
            adapter_pre_mlp=True,
            classifier = classifier,
            **kwargs,
        )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load('/hahmwj/ViT-B-16.pt', map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model
