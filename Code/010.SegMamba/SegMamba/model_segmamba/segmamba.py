# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba.mamba_ssm.modules.mamba_simple import Mamba      # from mamba_ssm import Mamba
import torch.nn.functional as F 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
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

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,      # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices, # 用于inter-wise direction的参数
        )
    
    def forward(self, x):       # [B,C,D,H,W]
        B, C = x.shape[:2]      # B为batch sieze，C为channel数，比如CT为1，但是经过stem卷积之后就不是了，而是48，96，192，384
        x_skip = x              # x_skip用作残差连接的部分
        assert C == self.dim
        n_tokens = x.shape[2:].numel()      # 获得patch num, 这是int类型的，也就是n_token=D*H*W
        img_dims = x.shape[2:]              # 也是patch num，是torch.Size类型的,并且是[D,H,W]用于后续恢复维度
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)        # [B,C,D,H,W]->[B,C,nToken]->[B,nToken,C]
        x_norm = self.norm(x_flat)          # 这是layernorm，维度是dim，对应的是C，也就是embed_dim的维度
        x_mamba = self.mamba(x_norm)        # 扔到mamba里面

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)        # [B,P,C]->[B,C,P]
        out = out + x_skip      # 残差
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):             # 在代码里面默认 mlp_dim = 2*hidden_size
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)       # 一步点积  [batch size, patch num, embed_dim*2]
        self.act = nn.GELU()                                # 非线性话
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)       # 一步点积  [batch size, patch num, embed_dim]

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        """
        这个GSC非常容易理解，按照Fig.3里面的图片进行就可以了
        Args:
            in_channles: 输入的维度数是多少，output也是多少，所以就一个参数
        """
        super().__init__()
        self.proj = nn.Conv3d(in_channles, in_channles, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        """
        Args:
            in_chans: 输入的channel为多少，默认为1，例如CT图像就是1
            depths: list，表示每一层ToM()里面有多少个Mmaba结构
            dims: list，表示每一次的下采样downsampling之后为多少，第一个是stem CNN输出的结果
            drop_path_rate:
            layer_scale_init_value:
            out_indices:用于索引的，就是Unet需要skip连接，所以需要记录下来，用这个东西记录，所以是非常天真无邪的0123
        """
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        stem = nn.Sequential(       # stem作为大卷积核，padding为3大小不变，输出的为dim[0]，
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),     # 使用kernel size=stride=2来进行减少大小，dim为list里面的下一个，来加深维度
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices      # 用于索引的，就是Unet需要skip连接，所以需要记录下来，用这个东西记录

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)        # 先下采样，可以把Stem CNN层也看做一个下采样
            x = self.gscs[i](x)                     # 然后GSC
            x = self.stages[i](x)                   # stage是Mamba layer

            if i in self.out_indices:               # 如果i在out_indices里面，我觉得是用indice表示几层来一个输出，所以我感觉这个out_indice的默认值有点不对
                # 获得self里面叫做norm{i}的，但是为什么不直接调用self.add_module里面的东西呢？
                # 或者说，如果self.add_module是一个特殊的用法，那为什么不把这个换成一个nn.Module去装四个呢？
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)               # 进行instance norm
                x_out = self.mlps[i](x_out)         # 进行mlp channel，也就是TSMamba的最后一个部分
                outs.append(x_out)                  # 把这个结果记录下来

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,                         # 默认输入的channel为1，也就是类似于CT图像的channel为1
        out_chans=13,                       #
        depths=[2, 2, 2, 2],                # 用为MambaEncoder里面的depths
        feat_size=[48, 96, 192, 384],       # 用于标注每一次downsampling的维度变化,而这个就是每一层的embed dim
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,             # 其实就是最后一层输出的Chanel的数量，实际上和feat_size弄到一起其实更好，768=2*384
        norm_name="instance",               #
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,           # 这个就是encoder这边的主体
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        self.encoder1 = UnetrBasicBlock(            # 其实就是UNETR中的block
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,                    # 这里的res_block默认为true，所以事实上是一个残差的block
        )                                           # 对应的是Fig.2里面的Res-block
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

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)           # [B，C,D,H,W]->
        enc1 = self.encoder1(x_in)      # [B,48, D/2, H/2, W/2]
        x2 = outs[0]
        enc2 = self.encoder2(x2)        # [B,96, D/4...]
        x3 = outs[1]
        enc3 = self.encoder3(x3)        # [B,192, D/8...]
        x4 = outs[2]
        enc4 = self.encoder4(x4)        # [B,384, D/16...]
        enc_hidden = self.encoder5(outs[3])     # [B,768, D/32..]
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
                
        return self.out(out)
    
