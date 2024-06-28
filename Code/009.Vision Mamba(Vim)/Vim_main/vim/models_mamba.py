# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten=True):
        """
        img_size: int 为图片的大小
        patch_size : int 为图片的patch size
        stride : 这个适用于把patch这个 16*16*3->embed_dim，所以他的大小理论上应该和patch_size一样大就可以了，不知道为什么还要再来一个，可能是作为了超参数可以调整，万一有人就是不想让他们一样
        in_channel int : 作为输入图像的chaneel数量
        embed_dim : 相当于词嵌入的维度,这里789没有用，在Vim里面的参数是187左右忘记了
        norm_layer : 类似于NormLayer/RMSNorm层传进来
        flatten ： 把那个压平，这个竟然还是可选的，我以为是必须的
        """
        super().__init__()
        img_size = to_2tuple(img_size)  # 转成tuple的形式
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape  # [B,C,H,W] == [batch size, channel(RGB), height, width]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # 进行卷积变成词嵌入维度 [B,C,H,W]->[B,embedding_dim, num_patches in one column, num_patches in one line]
        if self.flatten:
            x = x.flatten(2).transpose(1,2)  # [B, embed_dim, np, np]->[B,embed_dim,num_patches]->[b,num_patches, embed_dim]
        x = self.norm(x)  # 然后经过一个norm层，完成patch embedding，和VIsion transformer类似
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,                           # embedding dim的维度
                 mixer_cls,                     # 理论上是partial的object，但是调用之后会变成mamba class->mamaba object，
                                                # 如果要知道mixer_cls是什么，要去make_block看partial在干嘛
                 norm_cls=nn.LayerNorm,         #
                 fused_add_norm=False,          #
                 residual_in_fp32=False,        #
                 drop_path=0.,                  # drop path的比率
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
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)                 # 很神奇的传入一个dim就变成了mamba object了
        self.norm = norm_cls(dim)                   # 变成了一个norm层的object
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:                     # 如果需要这个 fused_add_norm，就需要确定self.norm必须是layernorm或者rmsnorm的一种
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).    hidden state是每一层的输出，第0层的patch encoding
            residual: hidden_states = Mixer(LN(residual))

            对于这两个参数的猜测，就是由于他们的架构导致的一个不是很合理的形式出现在这里
            由于Vim的作者修改了原mamba里面的代码，然后把Vim架构图里面的残差连接并没有写到修改过后的mamba的里面，所以需要在这个地方把hidden_state和residual给处理一下
            这里我的理解是：
                hidden state作为上一个block的输出，在Vim的架构图里面是输出之后没有经过最后一个与残差的加号的的那个（也就是每一个mamba不包括残差线）
                residual是上上一个block加之后的输出，也就是上一个block的输入
            所以在这个注释里面会有 hidden_states = Mixer(LN(residual)) 这个东西的存在，由于第一个blcok没有residual的存在，所以会有None的选择

            而代码中的fused_add_norm_fn这么看起来就是进行更新hidden_state和residual的，
                residual=hidden_stat
                然后 hidden_state可能要经过一个norm，就是Vim架构图里面的那个Norm

            这个地方没有非常非常具体的看，但是从作者的注释，改写的mamba源码中可以大概推理出这些内容，所以看起来这个地方乱七八糟的
            并且按照这个架构，那么最后一层的输出是不需要残差连接的，然后直接进行预测，根据cls或者avg pooling之类的，这个还没有看到
        """
        if not self.fused_add_norm:
            # 如果不需要fused add norm，主要是没有norm
            if residual is None:
                # 如果没有residual，代表是第一个block，所以直接让residual=hidden_state
                residual = hidden_states
            else:
                # 不然就是执行结构图里面的最后一步加
                residual = residual + self.drop_path(hidden_states)
            # 加完了之后，进行一步这个norm，就是这个block里面的hidden states了
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            # rms_norm_fn(还有layer_norm_fn)在mamba_ssm.ops.triton.layernorm里面
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # 对着hidden state进行一个mamba模块，得到输出
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,                # d_model相当于patch的embedding_dim，相当于NLP里面的词嵌入长度
        ssm_cfg=None,           # 在Vim-main.mamba_1p1p1.mamba_ssm.models.comfig_mamba.py里面有一个@dataclass的装饰器，
                                # 里面的参数作为默认参数可以用来配置这个
        norm_epsilon=1e-5,      # RMSNorm的eps
        drop_path=0.,           # drop path的比率
        rms_norm=False,         # 是否使用RMSNorm
        residual_in_fp32=False,
        fused_add_norm=False,   # fuse add norm指 有没有残差，如果有的话，就需要先add，再norm
        layer_idx=None,         # 层的编号
        device=None,
        dtype=None,
        if_bimamba=False,       # vision mamba提出了要forward backward两个方向进行一下 ssm那个通路的计算，这个是是否要需要多方向（双方向）
        bimamba_type="none",    # 这个我的理解是bimamba的类型是什么，为了方便修改后续的工作，比如说提出了四通道的，那么改一下这个就可以了 我猜的，后面过来修改一下这个注释
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:  # 如果需要多方向（在Vim中就是双方向），那么就把bimamba_type修改为v1
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # 相当于做了一个mamba出来，partial直接ctrl+左键点进去看，是一个包
    mixer_cls = partial(
        Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs
    )
    # 一个用于norm的
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,                            # embedding dim的维度长
        mixer_cls,                          # mamba模型类
        norm_cls=norm_cls,                  # norm层
        drop_path=drop_path,                # drop path的比率
        fused_add_norm=fused_add_norm,      #
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=224,                  # 图片大小
                 patch_size=16,                 # patch大小
                 stride=16,                     # stride大小，这个作为参数说明 一种变体不是那么严格的一个patch一个patch，可以有重叠
                 depth=24,                      # 多少层，
                 embed_dim=192,                 # embed_dim为多少，如果按照16*16*3=768，这还压缩了，竟然变小了
                 channels=3,                    # 图片通道数，正常RGB三通道就是3
                 num_classes=1000,              # Vim原文是用的imageNet-1k进行训练的，所以是1000
                 ssm_cfg=None,                  # ssm-cfg，可以填入config_mamba
                 drop_rate=0.,                  # drop rate比率，这个会用在位置编码里面的dropout，其他地方没有用到
                 drop_path_rate=0.1,            # drop path rate，在Block里面 对residual里面用
                 norm_epsilon: float = 1e-5,    # RMSNorm里面的eps
                 rms_norm: bool = False,        # True代表使用RMSNorm，False代表使用的是layerNorm
                 initializer_cfg=None,
                 fused_add_norm=False,          # 是否要进行fused add norm，在Block里面体现这个
                 residual_in_fp32=False,        # residual使用什么精度保存
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,              # 没看明白，是rope.VisionRotaryEmbeddingFast里面需要的一个参数
                 if_bidirectional=False,        # 是否使用双向，这个确实有用
                 final_pool_type='none',        #
                 if_abs_pos_embed=False,        # 是否使用绝对位置编码
                 if_rope=False,                 # if rope大概是要反转的sequence的意思，这个有点像是总开关，如果这个是false，那么永远不会进行反转类似的操作
                 if_rope_residual=False,        # if rope residual是要不要反转residual的意思
                 flip_img_sequences_ratio=-1.,  # 这个ratio会决定另一种个反转的东西
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,            # 是否使用cls token
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,    # 是否使用两个cls token，head一个，tail一个
                 use_middle_cls_token=False,    # 如果只是用一个cls token，那么是否要把这个cls token放到中间去
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs    把这个factory_kwargs加进去
        kwargs.update(factory_kwargs)
        super().__init__()
        # 把对应的参数用self保存下来
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # patch embedding层，使用的是上面定义的patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 如果需要有cls token，可以不用，如果不用就是avg pooling
        if if_cls_token:
            # 这个double代表着 要在所有token的最前面和最后面都加上一个cls token，所以现在有两个，估计是考虑到了这种时序性
            if use_double_cls_token:
                # 创造出对应的可训练参数tensor，这里的大小为[1,1,embed_dim]而不是[Batch size, 1, embed_dim]是因为
                # 在这个地方 __init__里面还不知道batch size是多少，所以先造出1，然后在forward()里面扩张到B
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2     # 修改number of token
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1   # 源码就注释掉了，不过这个无所谓，因为默认参数就是1

        # 使用绝对位置编码
        if if_abs_pos_embed:
            # 和上面一样，Batch size的位置设置成了1，所以后面要扩充，length为num_patch加上多出来的token的数量
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2          # embed_dim整除2的维度
            hw_seq_len = img_size // patch_size     # 一行或者一列有多少个patch
            self.rope = VisionRotaryEmbeddingFast(  # 在repo.py里面的
                dim=half_head_dim,                  # embed_dim除以2的维度
                pt_seq_len=pt_hw_seq_len,           # 不清楚，默认是14
                ft_seq_len=hw_seq_len               # 一行有多少个patch
            )

        # 这个就是预测层了，num_feature==embed_dim
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        # 这里的递减还是随着层的，如果按照一次拿两个层来做，前向和反向的权重递减还不对
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()  drop path的比率是每一层都不一样的，第一层为0
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head output head的norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        # 进行初始化，使用的timm.models.layers.trunc_normal_ 正态分布的初始化
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):      # 在这里写的参数名字中，不会进行权重衰减
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False,
                         if_random_token_rank=False):
        """
        x为输入的图片，[batch size, channel, height, width]
        inference_params 不知道是什么
        if_random_cls_token_position，在只是用一个cls token的时候，如果这个参数是true，那么就会把这个cls token随机插入一个位置中
        if random token rank:
        """
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        # 先经过patch embedding [batch, chanel, Height, width]->[batch, number patches, embed dim]
        x = self.patch_embed(x)
        B, M, _ = x.shape

        # 如果使用cls token，里面有一些子情况，例如使用两个cls token，一个cls token的位置在哪里
        if self.if_cls_token:
            if self.use_double_cls_token:   # 如果使用两个cls token，也就是head和tail
                cls_token_head = self.cls_token_head.expand(B, -1, -1)      # 扩充为Batch
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)      # 扩充为Batch
                token_position = [0, M + 1]                 # cls token position的位置是0和M+1放在一个list里面存折，后面会用到
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)   # 使用cat，把head, x, tail给拼接起来
                M = x.shape[1]                              # M为现在num_patches的大小，理论上是图片patch+2的大小在这种情况下
            else:                   # 另一个形式是 使用 一个cls token
                if self.use_middle_cls_token:   # 使用一个cls token，还可以把这个cls token放到中间
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                # 使用一个cls token，如果这个参数为true，就会把随机把cls token扔到某一个位置上，这个参数是在forward()里面传入的，不是__init__里面传入的
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    # 如果不是把一个cls token插入到middle也不是随机插入，那就是插入到head中，就是这里进行的
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                # 最后获得num_patch的长度,其实就是 img_num_patch+1
                M = x.shape[1]

        # 如果使用绝对位置编码
        if self.if_abs_pos_embed:
            # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
            #     x = x + self.pos_embed
            # else:
            #     pos_embed = interpolate_pos_embed_online(
            #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
            #             )
            x = x + self.pos_embed      # 加上位置编码
            x = self.pos_drop(x)        # 这个是用dropout进行的

        # 用于随机把token的位置打乱，但是一个cls token里面有这个选项了，这个的作用倒是有点奇怪，不过默认为false
        if if_random_token_rank:
            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in
                                      range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)


        # 先把image sequence是否反转了设置成falsse，也就是刚开始是不会反转image sequence的，但是在满足下面这个条件的时候就会反转
        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])     # 对着第1维进行反转操作
            if_flip_img_sequences = True

        # mamba impl        进行mamba里面的计算了
        residual = None     # residual只有在第一层的时候是none，
        hidden_states = x   # hidden state第一层前面就是输入的patch embedding [b, num_patch, embed_dim]，这里的num_patch是
        if not self.if_bidirectional:   # 如果不进行Vim里面提出的双向的操作
            # 只有单向ssm的时候，才会有要不要反转的问题，如果是双向里面，就没有这些代码，只有if_rope
            for layer in self.layers:   # 这个就是mamba里面的每一层

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:          # 如果if_flip_img_sequence，那么在经过了if_rope之后会在经过一遍这个fiip
                    hidden_states = hidden_states.flip([1])         # 有点不太懂，不过这里都是默认为false的
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        else:       # 这个else分支是使用Vim里面提出的双向操作
            # get two layers in a single for-loop
            # 这里的双向操作是通过一下拿两个层，一个层进行正向的，一个层进行反向的，所以这里n_layer的默认值为24是mamba的两倍
            # 这样可以在代码上偷懒，直接使用mamba的代码就可以了，
            # 但是但是有一点点不太对，如果是两个block分别进行前向和反向，那么两个block的映射层之类的是不一样的，
            # 但是从论文架构图中可以看到，映射曾应该是一样的，所以我感觉这里就是作者偷懒了，
            # 事实上应该从mamba里面的代码修改具体细节的代码，或者前向和反向都经过同一个层就可以了，不清楚为什么要这么做
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]),
                    inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        # 是否进行fused add norm，相当于最后一个block输出之后的最后一个fuse，所以说代码架构有点烂
        if not self.fused_add_norm:
            if residual is None:    # 有none是因为如果只有一层呢
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:   # 有cls token的东西
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]
        # 无cls token返回的情况
        if self.final_pool_type == 'none':      # 如果是none就把最后一维返回
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':    # 如果是mean，就把所有的进行平均，然后返回
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':     # 如果是max，就把这个返回，在forward()里面进行别的操作
            return hidden_states
        elif self.final_pool_type == 'all':     # 如果是all，那就真是all，全部都返回
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
                if_random_token_rank=False):
        # 第一步就是把x输入到forward_features()里面，这里面就会把所有的Vim block遍历，然后输出的是最后一个block输出的feature
        x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position,
                                  if_random_token_rank=if_random_token_rank)
        if return_features:     # 如果 forward()里面的参数return_features为true，就把这些feature返回，而不经过最后的MLP
            return x
        x = self.head(x)        # 经过最后的预测的MLP层      [B, num_patch, embed_dim]->[B,L,]
        if self.final_pool_type == 'max':   # 如果final_pool_type是max，就会使用max，把[B,L,embed]
            x = x.max(dim=1)[0]             # 返回了之后，经过线性层之后，然后那一个大就返回那一个
        return x


@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                             **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                              **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
