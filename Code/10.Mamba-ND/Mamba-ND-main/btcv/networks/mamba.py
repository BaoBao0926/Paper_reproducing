# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmcv.cnn import build_norm_layer
from mmengine.utils import to_2tuple
from typing import Tuple
import torch.nn.functional as F

MambaND = nn.Identity # TODO: Cleanup and release 
# from .mamband import MambaND
from mmpretrain.models.utils import (MultiheadAttention, SwiGLUFFNFused, build_norm_layer, resize_pos_embed, to_2tuple)

def resize_pos_embed(pos_embed: torch.Tensor,
                     src_shape: Tuple[int],
                     dst_shape: Tuple[int],
                     mode: str = 'trilinear',
                     num_extra_tokens: int = 1) -> torch.Tensor:
    """Resize pos_embed weights. 就是在之后的步骤里面，应该回存在着有不同大小的矩阵 都要进行位置编码，所以使用这个方法，
        可以把前一个矩阵的位置编码缩放成后一个矩阵的位置编码，整体思路是这样的。其中cls token的位置编码不需要缩放，所以先取出来，然后把剩下位置编码
        缩放，然后在拼接起来
    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape [1, L, C]. 这里的位置编码是已经flatten的
        src_shape (tuple): The resolution of downsampled origin training image, in format (T, H, W).
        dst_shape (tuple): The resolution of downsampled new training image, in format (T, H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest','linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'trilinear'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token. Defaults to 1.
    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    # 如果采样后的大小与采样前的大小是一样的，那么就直接返回位置编码parameter即可，如果不是基本的理念是通过F.interpolate给扩张或者收缩成原来
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1] \
            and src_shape[2] == dst_shape[2]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'      # .ndim是指有几个维度
    _, L, C = pos_embed.shape
    src_t, src_h, src_w = src_shape
    assert L == src_t * src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_t}*{src_h}*{src_w}+{num_extra_tokens}).' \
        'Please check the `img_size` argument.'
    # 假设cls token在最前面，把cls toekn的取出来，torch的默认语法是，多维的如果只写两个维度，就会默认对后面两个维度进行操作，
    extra_tokens = pos_embed[:, :num_extra_tokens]  # [1, L, num_extra_token]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_t, src_h, src_w, C).permute(0, 4, 1, 2, 3) # [1, Channel, t,h,w]

    dst_weight = F.interpolate(     # 进行采样，会把src_weight，按照mode规则才养成dst_shape的大小，也就是采样后的大小
        src_weight, size=dst_shape, align_corners=False, mode=mode) #[1,C,T,H,W]->[1，C，T2,H2,W2]，T2H2W2指采样后的维度
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)   # [1,c,t,h,w]->[1,c,L]->[1,l,c]

    # return把extra_token的位置编码和dst_weight的位置编码，拼接到了一起
    return torch.cat((extra_tokens, dst_weight), dim=1)


# from .base_backbone import BaseBackbone
# 没有配置mamba-ssm，所以直接把mamba_ssm源文件拿过来，修改了一下路径
# from mamba_ssm import Mamba
# from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from btcv.mamba_ssm.modules.mamba_simple import Mamba
from btcv.mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

from torch import Tensor
from typing import Optional
from functools import partial
# from .ssm2d import Block2D,Mamba2D,SplitHead2D
Block2D = Mamba2D = SplitHead2D = None
from einops import rearrange
from mmengine.logging import MMLogger


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.
    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed after the feed forward layer. Defaults to 0
        attn_drop_rate (float): The drop out rate for attention output weights. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.   Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,                # embedding dim
                 num_heads,                 # 多头的数量
                 feedforward_channels,      #
                 layer_scale_init_value=0., # Init value of layer scale
                 drop_rate=0.,              # dropout的比率
                 attn_drop_rate=0.,         # attention输出的z要进行dropout的比率
                 drop_path_rate=0.,         # dropout path的比率
                 num_fcs=2,                 # 全连接层有几层，默认为两层
                 qkv_bias=True,             # qkv是否需要偏执
                 ffn_type='origin',         # 选择FFN的type，默认为origin，还有一个参数是'swiglu_fused'，这个方法出来的不知道是什么
                 act_cfg=dict(type='GELU'), #
                 norm_cfg=dict(type='LN'),  # 使用mmcv构建一个标准norm层，需要的参数norm_cfg
                 norm_cfg_2=dict(type='LN'),#
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        # 第一个参数为输入参数 默认为layernorm，第二个参数是输出通道数量为embedin dim
        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,          # 应该为内部一个token的维度为多大
            num_heads=num_heads,            # 要几个多头
            attn_drop=attn_drop_rate,       # attention输出的z，进行drop out的比率
            proj_drop=drop_rate,            # 用于FFN里面的drop比率
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),  # 这个是用于drop path的，但是是个字典，可能还有别的参数
            qkv_bias=qkv_bias,              # 映射qkv的是否需要偏执
            layer_scale_init_value=layer_scale_init_value   # Init value of layer scale
        )
        # 这里应该就是想要有两种不同的norm层，所以有如果有norm_cfg_2那么就用2的，不然就用1的
        self.ln2 = build_norm_layer(norm_cfg_2 or norm_cfg, self.embed_dims)

        if ffn_type == 'origin':        # 进行feed-forward
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,        # 有多少个线性层
                ffn_drop=drop_rate,     # drop rate为多少
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,        # 激活函数，输入一个dict
                layer_scale_init_value=layer_scale_init_value    # Init value of layer scale，默认为0
            )
        elif ffn_type == 'swiglu_fused':    # 不清楚里面的是什么
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value)
        else:
            raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # 先norm1，在attention
        x = self.ffn(self.ln2(x), identity=x)       # 先norm2，在ffn
        return x

from mmcv.cnn.bricks.drop import build_dropout

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,reverse=False,
        transpose=False, split_head=False,
        drop_path_rate=0.0, drop_rate=0.0,use_mlp=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()                          # 这里很多参数与Vision Mamba一致
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)                 # 用的都是partial()去搞得
        self.norm = norm_cls(dim)
        self.split_head = split_head                # 新的
        self.reverse = reverse                      # 新的
        self.transpose = transpose                  # 新的
        self.drop_path = build_dropout(             # 构建一个drop path
            dict(type='DropPath', drop_prob=drop_path_rate)
        )
        self.dropout = build_dropout(               # 构建一个drop out
            dict(type='Dropout', drop_prob=drop_rate)
        )
        if use_mlp:
            self.ffn = SwiGLUFFNFused(
                    embed_dims=dim,
                    feedforward_channels=int(dim*4),
                    layer_scale_init_value=0.0)
            self.ln2 = build_norm_layer(dict(type='LN'), dim)
        else:
            self.ffn = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        order='t l h w',
        shape=None,     # 这个参数不能为None，不然会报错
        skip=True,
        n_dim_pos=4
    ):
        r"""Pass the input through the encoder layer.
            整体结构与Vision Mamba一致
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            shape不能为None，必须要传入数，不然asset会报错
        """

        # 和vision mamba比起来，主要就是这个由于n_dim_pos造成的维度变化为blcok最大的不一样，也是原文如何进行scan变化的代码
        h = w = 0
        assert shape is not None
        t, l, h, w = shape
        if n_dim_pos != 4:
            # 如果n_dim_pos=1最后的结果是 [n*t, l*h*w, c],把第二项给第一项，第一项代表了batch size或者说 有多少个第二项，第二项代表了Length，第三项是embed dim
            # 可能的值有1，2，4,但是具体怎么划分，还和order相关
            order = order.split(' ')        # [t, l, h, w]
            assert len(order) == 4
            trunc_n = 4 - n_dim_pos         # trunc_n=0
            tgt_order = f"(n {' '.join(order[:trunc_n])}) ({' '.join(order[trunc_n:])}) c" # [n, t, l, h, w, c]
        else:
            tgt_order = f'n ({order}) c'    # [n, t*l*h*w, c]

        # [n, t*l*h*w, c]->[n,t*l*h*w,c](这个是默认变化，随n_dim_pos的变化而变化)
        hidden_states = rearrange(hidden_states, f'n (t l h w ) c -> {tgt_order}', t=t, l=l, h=h, w=w)

        if self.transpose:      # transpose的作用就是把h和w反转一下
            l = hidden_states.shape[1]      # l=t*l*h*w
            h = w = int(np.sqrt(l))         # h=w=根号l
            # assert h * w == l
            hidden_states = rearrange(hidden_states, 'n (h w) c -> n (w h) c', h=h, w=w)        # 相当于把img转90°
            if residual is not None:
                residual = rearrange(residual, 'n (h w) c -> n (w h) c', h=h, w=w)

        if self.reverse:        # 把第二个元素flip反转一下
            hidden_states = hidden_states.flip(1)
            if residual is not None:
                residual = residual.flip(1)

        if not self.fused_add_norm:
            hidden_states = self.norm(hidden_states)
            # 跟新一下residual，这部分residual被注释掉了，可能是因为fused_add_norm在false的情况，根本用不到residual
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if self.split_head:     # 不知道啥玩意，文章也没有说这是什么，默认值为None，假设不存在好了
                l = hidden_states.shape[1]
                h = w = int(np.sqrt(l))
                hidden_states = SplitHead2D.apply(hidden_states,4,h,w)
            if skip:    # 传入forward（）的参数，默认为True
                # 就是要不要进行参加，这一步其实就是计算了， mamba->dropout->drop path 然后在加上原来的
                hidden_states = hidden_states + self.drop_path(self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
            else:   # 如果skip为false，就不加上残差
                hidden_states = self.drop_path(self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
            if self.split_head:     # 不管这个spilt head，不知道什么东西
                hidden_states = SplitHead2D.apply(hidden_states,4,h,w)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states = self.drop_path(self.mixer(hidden_states, inference_params=inference_params))
        if self.ffn is not None:
            hidden_states = self.ffn(self.ln2(hidden_states),identity=hidden_states)
        if self.reverse:
            hidden_states = hidden_states.flip(1)
            if residual is not None:
                residual = residual.flip(1)

        # 算完上面的一大堆，然后把这个维度返回成为了 [n,t,l,h,w,c]
        hidden_states = rearrange(hidden_states,f'{tgt_order}->n (t l h w ) c ',t=t,l=l,h=h,w=w)
        # 然后把img转动回来，这样就又恢复成了原来的东西了
        if self.transpose:
            hidden_states = rearrange(hidden_states,'n (w h) c -> n (h w) c',h=h,w=w)
            if residual is not None:
                residual = rearrange(residual,'n (w h) c -> n (h w) c',h=h,w=w)
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    reverse=None,
    is_2d=False,
    drop_rate=0.1,
    drop_path_rate=0.1,
    use_mlp=False,
    transpose=False,
    split_head=False,
    use_nd=False,
    n_dim=3,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if use_nd:
        transpose = False
        reverse = False
        mixer_cls = partial(MambaND , layer_idx=layer_idx, n_dim=n_dim,**ssm_cfg, **factory_kwargs)
    mixer_cls = partial(Mamba2D if is_2d else Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if is_2d:
        block = Block2D(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            drop_rate=drop_rate,
            transpose=transpose,
            drop_path_rate=drop_path_rate,
        )
    else:
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            transpose=transpose,
            drop_rate=drop_rate,
            use_mlp=use_mlp,
            drop_path_rate=drop_path_rate,
            split_head=split_head,
        )
    block.layer_idx = layer_idx
    return block 


from mmengine.runner.checkpoint import _load_checkpoint
import re
from prettytable import PrettyTable
        # with_cls_token=False,
        # final_norm=False,
        # fused_add_norm=False,
        # # norm_cfg=dict(
        # #     type='GN',num_groups=4, eps=1e-6
        # # ),
        # d_state=16,
        # is_2d=False,
        # use_v2=False,
        # use_nd=False,
        # force_a2=False,
        # pretrained=pretrained,
        # use_mlp=False,

class VisionMamba(BaseModule):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"cls_token"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small', 'dinov2-s', 'dinov2-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
        **dict.fromkeys(
            ['dinov2-g', 'dinov2-giant'], {
                'embed_dims': 1536,
                'num_layers': 40,
                'num_heads': 24,
                'feedforward_channels': 6144
            }),
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='base',                   # 这个参数决定模型架构是怎么样的，是从上面的arch_zoo里面搞出来，等于什么也是对应的
                 img_size=224,                  # 图片大小
                 patch_size=16,                 # patch大小
                 patch_size_temporal=2,
                 in_channels=3,                 # 输入的channel数量，用于patch embedding那个地方
                 out_indices=-1,
                 drop_rate=0.,                  # dropout的比率
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),    # 两个不同的norm层，由于是mmcv里面的build_norm_layer，所以要为dict
                 norm_cfg_2=dict(type='LN', eps=1e-6),  # 两个不同的norm层，
                 final_norm=True,               # 最后一层的norm
                 out_type='cls_token',          # 必须在这里面 OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}
                 with_cls_token=True,           # 要不要cls token，一定程度上会影响out_type，这个为false的时候，out_type不可以为cls_token
                 frozen_stages=-1,
                 interpolate_mode='bicubic',    # 用于resize_pos_embed那个插入方式
                 layer_scale_init_value=0.,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,                # prenorm，在位置编码之后，整体进入mamba block之前的norm，用的是norm_cfg_1的参数
                 init_cfg=None,
                 num_frames: int = 16,
                 expand=None,
                 inflate_len=False,         # 不知道是什么意思，4(层数*4)，False(当use mlp为F时候层数*2，use_mlp为T，层数*1)和True(层数*3)
                 is_1d=False,
                 is_2d=True,
                 use_v2=False,
                 force_a2=False,
                 has_transpose=True,
                 fused_add_norm=True,
                 use_mlp=False,             # 会影响inflate_len里面的层数的多少
                 split_head=False,
                 pretrained=None,           # 是否使用预训练的权重
                 pretrained2d=True,
                 dt_scale=0.0,
                 dt_scale_tmp=0.0,
                 use_nd=False,
                 force_2d=False,
                 update_interval=None,
                 copy_weight=False,
                 factorization=None,
                 inlfate_policy=None,
                 n_dim_pos=4,               # 这个很重要的参数，默认为4
                 n_dim=3,
                 embed_dims=None,           # embed_dim为多少，这个优先级比arch里面的高
                 num_layers=None,           # 一共有多少层，这个为基础的一个层，如果为None，会用arch里面的层数，如果为数，会优先用这个的数
                 single_dir=False,
                 d_state=16
                 ):
        super(VisionMamba, self).__init__(init_cfg)
        self.force_2d = force_2d
        self.use_nd = use_nd
        self.inlfate_policy = inlfate_policy
        self.pretrained2d = pretrained2d
        self.pretrained = pretrained
        self.n_dim_pos = n_dim_pos
        self.factorization = factorization
        self.inflate_len = inflate_len
        self.update_interval = update_interval
        self.copy_weight = copy_weight

        # 使用与训练的权重，把self.init_cfg给赋值乐
        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # 如果arch是string，那么先把string全部小写，给self.arch_setting赋值为arch_zoo对应的那个
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:   # 要么就是arch是字典，直接赋值，然后至少要有essential key里面的这些参数
            essential_keys = {'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'}
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
        # 如果num_layer为数，把num_layer替换成这个，也就是说num_layer的优先级会比arch高的多
        if num_layers is not None:
            self.arch_settings['num_layers'] = num_layers
        # embed_dim也是优先直接赋值，其次在世arch里面的
        self.embed_dims = embed_dims or self.arch_settings['embed_dims']

        # 这里就很奇怪了，根据不同的infalte_len，把对应的层数乘以对应的数字
        if self.inflate_len == 4:
            self.num_layers = self.arch_settings['num_layers'] * 4
        elif self.inflate_len:
            self.num_layers = self.arch_settings['num_layers'] * 3
        else:
            self.num_layers = self.arch_settings['num_layers'] * (2 if not use_mlp else 1)

        # 赋值img_size和is_2d
        self.img_size = img_size
        self.is_2d = is_2d

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,            # in_channels用于patch embedding
            input_size=self.img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv3d',                 # 使用的是conv3D
            kernel_size=patch_size,
            stride=patch_size,                  # 与Vision Mamba不同，这里就是直接stride=path size乐
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)，不太懂，使用pre_norm来控制bias
            padding=(0, 0, 0),
            dilation=(1, 1, 1)
        )
            #         in_channels=in_channels,
            # embed_dims=embed_dims,
            # conv_type='Conv3d',
            # kernel_size=(tubelet_size, patch_size, patch_size),
            # stride=(tubelet_size, patch_size, patch_size),
            # padding=(0, 0, 0),
            # dilation=(1, 1, 1)
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)         # 构建patch Embed
        self.patch_resolution = self.patch_embed.init_out_size  # patch embed应该就是一个conv3d，从conv3d里面得到输出为多大
        # [C，H,W]默认HW一样大，所以第三个元素直接取第二个元素大小
        self.patch_resolution = (self.patch_resolution[0], self.patch_resolution[1], self.patch_resolution[1])
        num_patches = self.patch_resolution[0] * self.patch_resolution[1] * self.patch_resolution[1]   # 算有多少个patch
        self.is_1d = is_1d

        # Set out type  OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set cls token，设置要不要cls_toekn，如果不要cls_token，那么out_type不可为cls_toekn，如果是的话，会报错
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != 'cls_token':
            self.cls_token = None       # 不是cls_token，更新一下参数
            self.num_extra_tokens = 0
        else:
            raise ValueError(
                'with_cls_token must be True when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode        # 用于resize_pos_embed那个插入方式
        # self.pos_embed = # [1, total patch, embeding]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # 看不懂， out_indices默认为-1
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            # assert 0 <= out_indices[i] <= self.num_layers, \
            #     f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        ssm_cfg={"d_state":d_state}
        if use_v2 and is_2d:
            ssm_cfg['use_v2'] = use_v2
        if force_a2:
            ssm_cfg['force_a2'] = force_a2
        if dt_scale > 0:
            ssm_cfg['dt_scale'] = dt_scale
        if dt_scale_tmp > 0 and (i//2)%3==2:
            ssm_cfg['dt_scale'] = dt_scale
        if expand is not None:
            ssm_cfg['expand'] = expand
        # 开始设置层
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            #self.layers.append(TransformerEncoderLayer(**_layer_cfg))
            self.layers.append(
                create_block(
                d_model=self.embed_dims,
                ssm_cfg=ssm_cfg,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=True,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                reverse=(not split_head) and (not single_dir) and (i % 2) > 0,
                transpose=(not split_head) and has_transpose and (i % 4) >= 2,
                use_mlp=use_mlp,
                is_2d=is_2d,
                rms_norm=False,
                split_head=split_head,
                use_nd=self.use_nd,
                n_dim=n_dim,
                )
            )
        self.frozen_stages = frozen_stages
        # prenorm，在位置编码之后，整体进入mamba blcok之前的norm
        if pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = final_norm    # 最后一层的norm
        if self.out_type == 'avg_featmap':
            self.ln1 = nn.Identity()
            self.ln2 = build_norm_layer(norm_cfg_2, self.embed_dims)
        elif final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        # if self.out_type == 'avg_featmap':
        #     self.ln2 = build_norm_layer(norm_cfg_2, self.embed_dims)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        #super(VisionMamba, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            # Inflate 2D model into 3D model.
            self.inflate_weights(logger)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if (not self.with_cls_token
                and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1):
            # Remove cls token from state dict if it's not used.
            state_dict[name] = state_dict[name][:, 1:]
            ckpt_pos_embed_shape = state_dict[name].shape

        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        b, _, _, h, w = x.shape
        # h //= self.patch_size
        # w //= self.patch_size
        
        x, patch_resolution = self.patch_embed(x)
        patch_resolution = (patch_resolution[0], patch_resolution[1], patch_resolution[1])

        x = x + resize_pos_embed(
            self.pos_embed,         # [1, total patch, embedding dim]
            self.patch_resolution,  # 3D卷积之后的patch resolution
            patch_resolution,       # 这里是上面patch_embed返回之后的resolution，不知道为什么可能不一样
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)[:, self.num_extra_tokens:]  # 这里还没有加上cls token，所以要切片

        if self.is_2d:
            assert self.cls_token is None
            x = rearrange(x, 'n (h w) c-> n c h w', h=patch_resolution[0], w=patch_resolution[1])

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((x, cls_token), dim=1) # append last
        x = self.drop_after_pos(x)               # 这里dropout了一下

        x = self.pre_norm(x)

        outs = []
        residual = None
        orders = (
                't l h w',
                't l w h',
                'w h t l'
        )
        if self.is_1d:
            orders = (
                't l h w',
                't l h w',
                't l h w',
            )
        if self.force_2d:
            orders = (
                't l h w',
                't l w h',
                't l h w',
            )

        n_dim_pos = [self.n_dim_pos] * 3

        if self.factorization is not None:
            if self.factorization == 'hw_t':
                n_dim_pos = (2, 2, 4)
            elif self.factorization == 'h_w_t':
                n_dim_pos = (1, 1, 2)
        shape = (patch_resolution[0], 1, patch_resolution[1], patch_resolution[2])
        raw_x = 0
        if self.update_interval:
            raw_x = x
            for i, blk in enumerate(self.layers):
                # i=1,z=0,d=0   i=2,z=1,d=1     i=3,z=1,d=1，所以最后的d为 001122 001122
                z = i // 2
                d = z % len(orders)
                x = x + blk(raw_x, order=orders[d], shape=shape, skip=False, n_dim_pos=n_dim_pos[d])
                if (i+1) % self.update_interval == 0 or i == len(self.layers) - 1:
                    raw_x = x
                #x = raw_x
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.ln1(x)

                if i in self.out_indices:
                    outs.append(self._format_output(x, patch_resolution))
        else:
            for i, blk in enumerate(self.layers):
                z = i // 2
                d = z % len(orders)
                
                x = blk(x, order=orders[d], shape=shape, n_dim_pos=n_dim_pos[d])
                    
            # for i, layer in enumerate(self.layers):
            #     x,residual = layer(x,residual)

                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.ln1(x)

                if i in self.out_indices:
                    outs.append(x)
        return outs[-1], outs
    
    def count_parameters(self,model=None):
        if model is None:
            model = self
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        self.total_parms = total_params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model, the weight
        of swin2d models should be inflated to fit in the shapes of the
        3d counterpart.

        Args:
            logger (MMLogger): The logger used to print debugging information.
        """
        if not self.pretrained:
            return
        checkpoint = _load_checkpoint(self.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('backbone.',''):v for k,v in state_dict.items()}
        curr_state_dict = self.state_dict()
        if self.inflate_len:
            new_weights = {}
            for k,v in state_dict.items():
                if 'layer' in k:
                    i_layer = int(re.compile('layers.([0-9]+).').findall(k)[0])
                    # 0 1 2 3 x x 4 5 6 7 x x
                    n_blk = i_layer // 4
                    n_idx = i_layer % 4
                    new_idx = n_blk * 6 + n_idx
                    k1 = k.replace(f'layers.{i_layer}',f'layers.{new_idx}')
                    assert k1 not in new_weights
                    new_weights[k1] = v
                    if self.copy_weight:
                        if  n_idx in [2,3]:
                            k2 = k.replace(f'layers.{i_layer}',f'layers.{new_idx+2}')
                            new_weights[k2] = v
                else:
                    new_weights[k] = v
            state_dict = new_weights
        for k in curr_state_dict:
            if k in state_dict:
                if (shape_1:=curr_state_dict[k].shape) != (shape_2:=state_dict[k].shape):
                    if 'patch_embed' in k:
                        state_dict[k] = state_dict[k].unsqueeze(-3).repeat(1,1,shape_1[2],1,1) / shape_1[2]
                        assert state_dict[k].shape ==shape_1
                    elif 'pos_embed' in k:
                        old_len = state_dict[k].shape[1]
                        state_dict[k] = state_dict[k].repeat(1,self.patch_resolution[0],1) #/ self.patch_resolution[0]
                        idxes = torch.arange(self.patch_resolution[0]).view(1,-1,1).repeat(1,1,old_len).view(1,-1,1).float()
                        if self.inlfate_policy == 'cosine':
                            state_dict[k]  = state_dict[k]  * torch.cos(idxes / self.patch_resolution[0] * np.pi)
                        elif self.inlfate_policy == 'single':
                            state_dict[k]  = state_dict[k]  * (idxes ==( self.patch_resolution[0]//2))
                        assert state_dict[k].shape ==shape_1,(state_dict[k].shape,shape_1)
                    else:
                        print(k,shape_1,shape_2)
            else:
                print(k)
                #re.compile('')
        # delete relative_position_index since we always re-init it
        
        # bicubic interpolate relative_position_bias_table if not match
        # relative_position_bias_table_keys = [
        #     k for k in state_dict.keys() if 'relative_position_bias_table' in k
        # ]
        # for k in relative_position_bias_table_keys:
        #     relative_position_bias_table_pretrained = state_dict[k]
        #     relative_position_bias_table_current = self.state_dict()[k]
        #     L1, nH1 = relative_position_bias_table_pretrained.size()
        #     L2, nH2 = relative_position_bias_table_current.size()
        #     L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        #     wd = self.window_size[0]
        #     if nH1 != nH2:
        #         logger.warning(f'Error in loading {k}, passing')
        #     else:
        #         if L1 != L2:
        #             S1 = int(L1**0.5)
        #             relative_position_bias_table_pretrained_resized = \
        #                 torch.nn.functional.interpolate(
        #                     relative_position_bias_table_pretrained.permute(
        #                         1, 0).view(1, nH1, S1, S1),
        #                     size=(2 * self.window_size[1] - 1,
        #                           2 * self.window_size[2] - 1),
        #                     mode='bicubic')
        #             relative_position_bias_table_pretrained = \
        #                 relative_position_bias_table_pretrained_resized. \
        #                 view(nH2, L2).permute(1, 0)
        #     state_dict[k] = relative_position_bias_table_pretrained.repeat(
        #         2 * wd - 1, 1)

        # In the original swin2d checkpoint, the last layer of the
        # backbone is the norm layer, and the original attribute
        # name is `norm`. We changed it to `norm3` which means it
        # is the last norm layer of stage 4.
        # if hasattr(self, 'norm3'):
        #     state_dict['norm3.weight'] = state_dict['norm.weight']
        #     state_dict['norm3.bias'] = state_dict['norm.bias']
        #     del state_dict['norm.weight']
        #     del state_dict['norm.bias']

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, -1]
        if not self.is_2d:
            patch_token = x[:, self.num_extra_tokens:]
        else:
            patch_token = x
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            if self.is_2d:
                return patch_token
            else:
                return patch_token.reshape(B, *hw, -1).permute(0, 4, 1, 2,3)
        if self.out_type == 'avg_featmap':
            if self.is_2d:
                return self.ln2(patch_token.mean(dim=1))
            else:
                return self.ln2(patch_token.mean(dim=1))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers


if __name__ == "__main__":
    vm = VisionMamba(in_channels=10)
    img = torch.ones(2,10,224,224)
    output = vm(img)
    print(output.size())