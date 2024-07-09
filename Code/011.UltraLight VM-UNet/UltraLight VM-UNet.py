"""
source code: https://github.com/wurenkai/UltraLight-VM-UNet
In the Fig.2, the stage 6 should be passed into SAB CAB, but in fact in source code, it does not. I change this
The source code is very clear to read

I add more parameter
num_divided_chan, in the passage, it has four divided channel groups. I set this as hyper-parameter
is_same_mamba: I find four channel groups are all passed into one mamba. If this is False, four channel will be passed into corresponding four mamba
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from UltraLight_VM_UNet_main.mamba_ssm.modules.mamba_simple import Mamba


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, num_divided_chan=4, if_same_mamba=True):
        super(PVMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(self.input_dim)
        self.num_divided_chan = num_divided_chan
        self.if_same_mamba = if_same_mamba
        if if_same_mamba:
            self.mamba = Mamba(
                d_model=input_dim//self.num_divided_chan,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba_list = []
            for i in range(num_divided_chan):
                self.mamba_list.append(
                    Mamba(
                        d_model=input_dim // self.num_divided_chan,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                )
        self.proj = nn.Linear(self.input_dim, self.output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x:torch.Tensor):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, D = x.shape
        x_flat = x.reshape(B, C, H*D).transpose(-1, -2)         # [B, N_TOKEN, C]
        x_norm = self.norm(x_flat)

        x_list = list(torch.chunk(x_norm, self.num_divided_chan, dim=-1))
        # x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=-1)

        if self.if_same_mamba:
            for i in range(self.num_divided_chan):
                x_list[i] = self.mamba(x_list[i]) + self.skip_scale * x_list[i]
        else:
            for i in range(self.num_divided_chan):
                x_list[i] = self.mamba_list[i](x_list[i]) + self.skip_scale * x_list[i]

        x_mamba = torch.cat([i for i in x_list])

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, H, D)
        return out


class Channel_Att_Bridge(nn.Module):
    """CAB"""

    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list)    # 128+64
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        # I add this one
        self.att6 = nn.Linear(c_list_sum, c_list[5]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[5], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5, t6):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5),
                         self.avgpool(t6)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        att6 = self.sigmoid(self.att6(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
            att6 = att6.transpose(-1, -2).unsqueeze(-1).expand_as(t6)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
            att6 = att6.unsqueeze(-1).expand_as(t6)

        return att1, att2, att3, att4, att5, att6


class Spatial_Att_Bridge(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5, t6):
        t_list = [t1, t2, t3, t4, t5, t6]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4], att_list[5]


class SC_Att_Bridge(nn.Module):
    # 把上面的CAB SAB合并到一起，用的时候就可以直接调用这个类
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5, t6):
        r1, r2, r3, r4, r5, r6 = t1, t2, t3, t4, t5, t6

        satt1, satt2, satt3, satt4, satt5, satt6 = self.satt(t1, t2, t3, t4, t5, t6)
        t1, t2, t3, t4, t5, t6 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5, satt6*t6

        r1_, r2_, r3_, r4_, r5_, r6_ = t1, t2, t3, t4, t5, t6
        t1, t2, t3, t4, t5, t6 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5, t6+r6

        catt1, catt2, catt3, catt4, catt5, catt6 = self.catt(t1, t2, t3, t4, t5, t6)
        t1, t2, t3, t4, t5, t6 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5, catt6 * t6

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_, t6 + r5_


class UltraLight_VM_UNet(nn.Module):
    """
    num_classes很显然是有几个类别，默认为1是因为默认为ISIC2018数据集，是对皮肤病分割的，只有一个class
    input_channel为图片是几维度的，这里默认处理的是2D图片，这些数据集都不是3D的
    c_list为channel之后的大小，比如，第一个encoder输出之后是8，这里可以看到，为了节约参数，这里的channel都非常的小
    split_att
    bridge，是否要使用Fig.2中间的SAB->CAB
    """

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True, d_state=16, d_conv=4, expand=2, num_divided_chan=4,
                 if_same_mamba=True):

        super().__init__()

        self.bridge = bridge
        # encoder1-3是三个2d卷积，size为3，stride为1，padding为1，不改变resolution，输入维度是c_list[n]，输出维度是c_list[n+1]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )

        # encoder4-6是文章提出的PVMLayer，维度也是按照和上面一样的
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3], d_state=d_state, d_conv=d_conv, expand=expand,
                     num_divided_chan=num_divided_chan, if_same_mamba=if_same_mamba)
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4], d_state=d_state, d_conv=d_conv, expand=expand,
                     num_divided_chan=num_divided_chan, if_same_mamba=if_same_mamba)
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5], d_state=d_state, d_conv=d_conv, expand=expand,
                     num_divided_chan=num_divided_chan, if_same_mamba=if_same_mamba)
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4], d_state=d_state, d_conv=d_conv, expand=expand,
                     num_divided_chan=num_divided_chan, if_same_mamba=if_same_mamba)
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3], d_state=d_state, d_conv=d_conv, expand=expand,
                     num_divided_chan=num_divided_chan, if_same_mamba=if_same_mamba)
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2], d_state=d_state, d_conv=d_conv, expand=expand,
                     num_divided_chan=num_divided_chan, if_same_mamba=if_same_mamba)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.ebn6 = nn.GroupNorm(4, c_list[5])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)  # num_class标志着要输出多少个维度

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(F.max_pool2d(self.ebn6(self.encoder6(out)), 2, 2))
        t6 = out    # B, C5, H/64, W/64

        # 和Fig.2的图一样，把源代码的这个弄上去了

        if self.bridge:
            t1, t2, t3, t4, t5, t6 = self.scab(t1, t2, t3, t4, t5, t6)

        out = torch.add(out, t6)
        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32

        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16

        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8

        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4

        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2

        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2
        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)