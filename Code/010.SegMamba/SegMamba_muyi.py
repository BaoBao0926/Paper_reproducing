"""
source code is SegMamba
In SegMamba, it rewrite the code of Mamba and write the segmamba.py
In this python file, I will rewrite the Mamba and segmamba as toy version
"""


from __future__ import annotations
import torch.nn as nn
import torch
from functools import partial

from SegMamba.monai.networks.blocks.dynunet_block import UnetOutBlock
from SegMamba.monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from SegMamba.mamba.mamba_ssm.modules.mamba_simple import Mamba      # from mamba_ssm import Mamba
import torch.nn.functional as F


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:    # 这个mamba_inner_fn_no_out_proj是用到了的
    from SegMamba.mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from SegMamba.mamba.mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from SegMamba.mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



class Mamba(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
            conv_bias=False, bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type='v3', # 默认是none，但是Segmamba代码里面，如果不是v3会报错(assert了),所以这里我就直接v3了
            nslices=5
    ):
        factory_kwargs={"device":device, "dtype":dtype}
        super(Mamba, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)      # mamba内部处理的维度数是d_model的expand倍
        self.dt_rank = math.ceil(self.d_model/16) if dt_rank=="auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_index = layer_idx
        self.nslices = nslices
        self.bimamba = bimamba_type

        # 用于最刚开始的映射 ->x与z两个，每一个的维度是d_inner，d_inner=expand*d_model
        self.in_proj = nn.Linear(self.d_mmodel, self.d_inner*2, bias=bias)

        self.activation = "silu"
        self.act = nn.SiLU()

        # 这里是用于初始化
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.copy_(inv_dt)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # 接下来是用于ssm模块里面的东西，从segmamba代码里面来看，事实上三个方向的都是一样的，所以我们用一个list装起来，在forward里面修改具体的
        self.conv1d_list = []
        self.x_proj_list = []
        self.dt_proj_list = []
        self.A_log_list = []
        self.D_list = []
        for _ in range(3):
            A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                       "n -> d n",
                       d=self.d_inner
                       )
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
            self.A_log_list.append(A_log)
            D = nn.Parameter(torch.ones(self.d_inner, device=device))
            D._no_weight_decay = True
            self.D_list.append(D)
            # 用于ssm前面的那个1d的卷积，把相邻直接的token进行信息交互
            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs
            )
            self.conv1d_list.append(conv1d)
            # 用于把x映射dt，B,C
            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + 2 * self.d_state, bias=False, **factory_kwargs
            )
            self.x_proj_list.append(x_proj)
            # 用于把dt映射为A
            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.dt_proj_list.append(dt_proj)

            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        Args:
            hidden_states:  [B, L, D]
            inference_params:    [B, L ,D]
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        if self.use_fast_path and inference_params is None:     # 他的else我就不写了，因为一定要用这个if语句里面的东西
            # 这里就写一个v3的bimamba_type
            if self.bimamba_type == "v3":
                A_list = []
                for i in range(3):
                    A_list.append(-torch.exp(self.A_log_list[i].float()))
                out_list = []
                for i in range(3):
                    # 三个方向的处理
                    if i == 0:          # forward
                        xz_i = xz
                    elif i == 1:        # backward
                        xz_i = xz.flip([-1])
                    else:
                        # inter wise ward,in source code, I explain this operation. and I think it is not very good
                        xz_i = xz.chunk(self.nslices, dim=-1)
                        xz_i = torch.stack(xz_i, dim=-1)
                        xz_i = xz_i.flatten(-2)

                    out = mamba_inner_fn_no_out_proj(
                        xz_i,
                        self.conv1d_list[i].weight,
                        self.conv1d_list[i].bias,
                        self.x_proj_list[i].weight,
                        self.dt_proj_list[i].weight,
                        A_list[i],
                        None,
                        None,
                        self.D_list[i].float(),
                        delta_bias=self.dt_proj_list[i].bias.float(),
                        delta_softplus=True
                    )
                    if i == 2:
                        out = out.reshape(batch, self.d_inner, seqlen//self.nslices, self.nslices)
                        out = out.permute(0, 1, 3, 2).flatten(-2)
                    out_list.append(out)

                out = F.linear(rearrange(out_list[0]+out_list[1].flip([-1])+out_list[2], "b d l -> n l d"),
                           self.out_proj.weight, self.out_proj.bias)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__(),
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":         # [B, H, W, C]
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":      # [B, C, H, W]
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MlpChannel(nn.Module):

    def __init__(self, hidden_size, mlp_dim):
        super(MlpChannel, self).__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):

    def __init__(self, in_channels):
        super(GSC, self).__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=3, padding=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonlinear = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonlinear2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonlinear3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonlinear4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.nonlinear(self.norm(self.proj(x)))
        x1 = self.nonlinear2(self.norm2(self.proj2(x1)))
        x2 = self.nonlinear3(self.norm3(self.proj3(x1)))

        x = x1 + x2
        x = self.nonlinear4(self.norm4(self.proj4(x)))
        return x + x_residual


class SegMambaLayer(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super(SegMambaLayer, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices
        )

    def forward(self, x):   # [B,C,D,H,W]
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()      # n_tokens = D*H*W
        img_dims = x.shape[2:]              # img_dim: torch.size = [D, H, W]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        return out



class SegMambaEncoder(nn.Module):

    def __init__(self, in_chans=1,
                 depths=[2,2,2,2],
                 dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0,1,2,3]
                 ):
        super(SegMambaEncoder, self).__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])
            self.gscs.append(gsc)
            stage = nn.Sequential(
                *[SegMambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2*dims[i_layer]))

    def forward(self, x):
        outs = []
        for i in range(4):  # 4是因为我们知道只有四层，其实可以len(depths)
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)


class SegMamba(nn.Module):

    def __init__(self,
                 in_chans=1,
                 out_chans=13,
                 depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384],
                 drop_path_rate=0,
                 layer_scale_init_value=1e-6,
                 hidden_size=768,
                 norm_name="instance",
                 conv_block=True,
                 res_block=True,
                 spatial_dims=3
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = SegMambaEncoder(in_chans,  # 这个就是encoder这边的主体
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                )
        self.encoder1 = UnetrBasicBlock(  # 其实就是UNETR中的block
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,  # 这里的res_block默认为true，所以事实上是一个残差的block
        )  # 对应的是Fig.2里面的Res-block
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def forward(self, x):
        outs = self.vit(x)  # [B，C,D,H,W]->
        enc1 = self.encoder1(x)  # [B,48, D/2, H/2, W/2]
        x2 = outs[0]
        enc2 = self.encoder2(x2)  # [B,96, D/4...]
        x3 = outs[1]
        enc3 = self.encoder3(x3)  # [B,192, D/8...]
        x4 = outs[2]
        enc4 = self.encoder4(x4)  # [B,384, D/16...]
        enc_hidden = self.encoder5(outs[3])  # [B,768, D/32..]
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        return self.out(out)























































