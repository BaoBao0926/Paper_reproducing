"""
这里我想写一个玩具版本的Vision Mamba Vim，整体架构和官方代码基本一致
官方github仓库： https://github.com/hustvl/Vim
由于Vim把Mamba源码修改了一些然后封装了，所以必须要先去看Mamba的源码
我在官网Vim里面的代码写了很多注释，所以这个玩具版本里面就没有很多注释，可以对照着看下

我省略很多没有用的
1.写了双向ssm和不双向ssm
2.写了double cls token和一个cls token，没有写ramdom shuffle，没有写单cls token随机插入，没有rope那些
3.没有用partial()，因为我又点看不懂
4.简化了最后的返回
    4.1 如果有cls token的情况，直接返回token了，由于少了其他情况，所以少了一些情况处理
    4.2 如果没有cls token，按照源码进行处理
由于只是用来理解的，我在window里面写的，所以没有办法解决causal_conv1d里面的东西，所以不能检查究竟有没有问题，但是整体上来看是ok的
"""

from functools import partial

import torch
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from Vim_main.mamba_1p1p1.mamba_ssm.modules.mamba_simple import Mamba
from Vim_main.mamba_1p1p1.mamba_ssm.utils.generation import GenerationMixin
from Vim_main.mamba_1p1p1.mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from Vim_main.vim.rope import *
import random

try:
    from Vim_main.mamba_1p1p1.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224, patch_size=16, stride=16, in_channs=3, embed_dim=768,
                 norm_layer=None, flatten=True):
        super(PatchEmbedding, self).__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0]*self.patch_size[1]
        self.flatten = flatten

        self.projection_layer = nn.Conv2d(
            in_channels=in_channs,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x:Tensor):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size({H}*{W}) does not match the model({self.img_size[0]}*{self.img_size[1]})"
        x = self.projection_layer(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,                           # embedding dim的维度
                 mixer_cls,                     # 理论上是partial的object，但是调用之后会变成mamba class->mamaba object，
                                                # 在我这里我直接使用mamaba对象传进来使用
                 norm_cls=None,                 # 也修改一下，如果是None，就是默认为LayerNorm
                 # fused_add_norm=False,          # 不适用这个参数，默认为false，不写true的分支
                 residual_in_fp32=False,        # 这个有点用 给他写上
                 drop_path=0.,                  # drop path的比率
                 ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls
        self.norm = nn.LayerNorm(dim) if norm_cls is None else norm_cls
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor], inference_params=None):
        """
        这些参数在我得官方代码里面的注释我解释过了，由于我不知道inference params在干什么，所以我就照着写过来了
        这里需要先看一下官方代码注释里面我对这个架构代码的解释，才会有更明白一些,
        hidden_states: the sequence to the encoder layer (required).    hidden state是每一层的输出，第0层是patch encoding
        residual: hidden_states = Mixer(LN(residual))
        """

        # 更新一下
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)

        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """不清楚这个是用来干嘛的，就放在这里，不影响大部分的理解"""
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
        d_model,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        if_bimamba=False,
        bimamba_type='v1',
        layer_index=None
):
    if if_bimamba:
        bimamba_type = 'v1'
    mixer_cls = Mamba(d_model, layer_idx=layer_index, bimamba_type=bimamba_type)
    norm_cls = RMSNorm(d_model, norm_epsilon) if rms_norm else nn.LayerNorm(d_model)

    block = Block(
        dim=d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path
    )
    return block


# 两个初始化参数的函数，就直接抄下来了，不用太管
# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer,
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
                 img_size=224,
                 patch_size=16,
                 stride=16,
                 depth=24,      # 由于双向是一次用掉两个层,所以是2*12
                 embed_dim=192,
                 channels=3,
                 num_classes=1000,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float =1e-5,
                 rms_norm=False,
                 residual_in_fp32=False,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 use_double_cls_token=False,
                 # use_middle_cls_token=False       # 使用一个cls token的情况就不管了 看源码得了
                 ):

        super(VisionMamba, self).__init__()
        # 把对应的参数用self保存下来
        self.residual_in_fp32 = residual_in_fp32
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        # self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embedding = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, stride=stride, in_channs=channels, embed_dim=embed_dim
        )
        self.num_patch = self.patch_embedding.num_patches

        # 设置cls token和num of cls token
        if self.if_cls_token:
            if self.use_double_cls_token:   # 使用两个cls token
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:   # 使用一个cls token
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 1

        # 设置位置编码
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch+self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList([
            create_block(
                d_model=embed_dim,
                norm_epsilon=norm_epsilon,
                drop_path=inter_dpr[i],
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                layer_index=i
            ) for i in range(depth)
        ])

        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(embed_dim, eps=norm_epsilon)

        # 权重初始化-这里就是这样的子的，不想初始化可能对结果有点影响，但是不填影响理解，所以这块可以不看
        self.patch_embedding.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if self.if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if self.use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_weights)

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

    def forward(self, x, inference_params=None, return_feature=False):        # 其他那些什么random shuffle的参数都不管了
        x = self.forward_feature(x, inference_params=inference_params)

        if return_feature:
            return x    # 把feature返回

        x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x

    def forward_feature(self, x, inference_params=None):
        # patch embedding
        x = self.patch_embedding(x)
        B, M, _ = x.shape

        # 如果使用cls token里面的处理
        token_position = 0      # 加一个全局变量，事实上没有也一样
        if self.if_cls_token:
            if self.use_double_cls_token:       # 使用两个cls token的情况
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat([cls_token_head, x, cls_token_tail], dim=1)
                M = x.shape[1]
            else:
                cls_token = self.cls_token.expand(B, -1, -1)
                token_position = 0
                M = x.shape[1]

        # 使用绝对位置编码里面的操作
        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # 反转那些全部不写
        # 开始经过层里面的操作了
        residual = None
        hidden_states = x
        if not self.if_bidirectional:   # 只进行单向ssm
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
        else:   # 进行双向ssm
            # 这里是源码里面的写法-我觉得有问题
            for i in range(len(self.layers)//2):
                hidden_states_f, residual_f = self.layers[i*2](hidden_states, residual, inference_params)
                hidden_states_b, residual_b = self.layers[i*2+1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                    inference_params=inference_params)

                hidden_states = hidden_states_b.flip([1]) + hidden_states_f
                residual = residual_f + residual_b.flip([1])
            # 这个是我认为的代码，应该是对着一个block跑一遍正向和反向，而不是对着两个block，这样才比较对，不过更好的方法，应该是在mamba里面进行修改可能会更好一些
            # for layer in self.layers:
            #     hidden_states_f, residual_f = layer(hidden_states, residual, inference_params)
            #     hidden_states_b, residual_b = layer(hidden_states.flip([1]), residual.flip([1]), inference_params)
            #
            #     hidden_states = hidden_states_b.flip([1]) + hidden_states_f
            #     residual = residual_f + residual_b.flip([1])

            # 这里相当于最后一个block输出之后的 那个fuse，所以我说代码架构有点烂
            if residual is None:    # 有none是因为如果只有一层呢
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            # 开始返回东西
            if self.if_cls_token:
                if self.use_double_cls_token:
                    # 两个double是返回的是 两个cls token相加 除以2的
                    return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
                else:   # 如果是一个cls token
                    # 如果是一个cls token， 直接返回这个token 就行了
                    return hidden_states[:, token_position, :]
            else:   # 没有cls的情况
                if self.final_pool_type == 'none':      # none返回最后一个，其实没有啥用
                    return hidden_states[:, -1, :]
                if self.final_pool_type == 'mean':      # mean代表把所有的平均返回，我感觉这个更好一点，比none好多了
                    return hidden_states.mean(dim=1)
                if self.final_pool_type == 'max' and self.final_pool_type == 'all':
                    return hidden_states

if __name__ == "__main__":
    print("test")
    vm = VisionMamba(img_size=224, depth=2, channels=3, if_cls_token=True)
    img = torch.ones(10, 3, 224, 224)
    output = vm(img)
    print(output.size())




















