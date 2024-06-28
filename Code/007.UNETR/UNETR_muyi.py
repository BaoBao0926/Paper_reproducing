import torch
import ViT_Muyi
from torch import nn as nn


class SingleDeconv3DBlock(nn.Module):
    """
    用于3d图像里面的反卷积，图像大小会扩大一倍，所以里面的kernel size和stride都是2
    in_channel和out_channel作为参数自己定
    一个基本单位
    """
    def __init__(self, in_planes, out_planes):
        super(SingleDeconv3DBlock, self).__init__()
        self.block = nn.ConvTranspose3d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0
        )

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    """
    因为在实现的部分中，会有3*3*3大小的卷积核，也有2*2*2的卷积核，所以kernel_size可以自己定
    这个相当于一个基本的block
    """
    def __init__(self, in_plances, out_planes, kernel_size):
        super(SingleConv3DBlock, self).__init__()
        self.block = nn.Conv3d(
            in_channels=in_plances,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2)
        )

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    """
    一个完整的3D卷积，带有batchnorm和relu的
    用到的一个组合
    """
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(Conv3DBlock, self).__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            # nn.ReLU(True)     # 不知道为什么这里来了个True,所以我选择单独弄出来
        )
        self.r = nn.ReLU()

    def forward(self, x):
        return self.r(self.block(x))


class Deconv3DBlock(nn.Module):
    """
    一个完整的Decov，是一个2*2*2的反卷积，和一个3*3*3的3D卷积， BatchNorm和Relu
    用到的组合
    """
    def __init__(self, in_planed, out_planes, kernel_size=3):
        super(Deconv3DBlock, self).__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planed, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class UNETR(nn.Module):
    """
    感觉这里的模型搭建和swin-transofrmer Vit比，就是有点简单，但是让人很容易看得懂
    """
    def __init__(self, *, img_shape=(128, 128, 128), in_channel, output_dim=3,
                 embed_dim=768, patch_size=16, num_heads=12, dropout=0.1, depth=12, mlp_dim):
        """
        img_shape 指的是，输入的一个3D图片的大小是什么，第一个是channel数量，第二三是宽高
        input_dim 为 输入
        """
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_size = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
        #              mlp_dim, channel=3, dim_head=64, dropout=0., pool="mean"):

        self.vit = ViT_Muyi.ViT(
            image_size=img_shape[1],
            patch_size=patch_size,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            channel=in_channel,
            cube_size=img_shape
        )

        # U-Net Decoder
        # 图中蓝色的叫decoder
        self.decoderz3 = nn.Sequential(
            Deconv3DBlock(768, 768),
            Deconv3DBlock(768, 500),
            Deconv3DBlock(500, 128)
        )
        self.decoderz6 = nn.Sequential(
            Deconv3DBlock(768, 512),
            Deconv3DBlock(512, 256)
        )
        self.decoderz9 = Deconv3DBlock(768, 512)

        # 图中绿色的单独的deco
        self.decov12 = SingleDeconv3DBlock(768, 512)
        self.decov9 = SingleDeconv3DBlock(512, 256)
        self.decov6 = SingleDeconv3DBlock(256, 128)
        self.decov3 = SingleDeconv3DBlock(128, 64)

        # 图中黄色的部分
        self.conv9 = nn.Sequential(Conv3DBlock(1024, 512), Conv3DBlock(512, 512))
        self.conv6 = nn.Sequential(Conv3DBlock(512, 256), Conv3DBlock(256, 256))
        self.conv3 = nn.Sequential(Conv3DBlock(256, 128), Conv3DBlock(128, 128))
        self.conv0 = nn.Sequential(Conv3DBlock(in_channel, 64), Conv3DBlock(64, 64))

        # 图中灰色部分
        self.output = nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )

    def forward(self, x):
        
        result = self.vit(x)

        # x  [10, 4, 128, 128, 128]
        # zx [10, 512, 768]
        z3, z6, z9, z12 = result[2], result[5], result[8], result[11]

        # 这里的做法是，把Vit输出的[batch_size, patch_number, patch_embedding] ==[10, 512, 768]
        # -> [batch_size, patch_embedding, 剩下的是512（8的三次方）所以接下来三个维度都是8] == [10, 768, 8, 8, 8]
        # 现在这个[10, 768, 8, 8, 8]是W/16, H/16, D/16,通过一步步卷积和反卷积，得到最后的结果
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)     # [10, 768, W/16, 8, 8]
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)     # [10, 768, W/16, 8, 8]
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)     # [10, 768, W/16, 8, 8]
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)   # [10, 768, W/16, 8, 8]

        print("11")
        z12 = self.decov12(z12)     # [10, 512, H/8, 16, 16]
        z9 = self.decoderz9(z9)     # [10, 512, H/8, 16, 16]
        print("22")
        z9 = torch.cat([z9, z12], dim=1)    # [10, 1024, H/8, 16, 16]
        z9 = self.conv9(z9)         # [10, 512, H/8, 16, 16]
        print("3")
        z9 = self.decov9(z9)        # [10, 256, H/4, 32, 32]
        z6 = self.decoderz6(z6)      # [10, 256, H/4, 32, 32]
        print("4")
        z6 = torch.cat([z6, z9], dim=1)     # [10, 512, H/4, 32, 32]
        z6 = self.decov6(self.conv6(z6))    # [10, 128, H/2, 64, 64]
        print("5")
        z3 = self.decoderz3(z3)     # [10, 128, H/2, 64, 64]
        z3 = torch.cat([z3, z6], dim=1)     # [10, 256, H/2, 64, 64]
        print("6")
        z3 = self.decov3(self.conv3(z3))    # [10, 64,  H, 128, 128]
        z0 = self.conv0(x)         # [10, 64, H, 128, 128]
        print("7")
        z0 = torch.cat([z0, z3], dim=1)     # [10, 128, H, 128, 128]

        print("22")
        output = self.output(z0)

        return output



if __name__ == "__main__":

    image_shape = (128, 128, 128)
    in_channel = 4

    v = UNETR(
        img_shape=image_shape,
        output_dim=3,
        embed_dim=768,
        patch_size=16,
        num_heads=12,
        dropout=0.1,
        in_channel=in_channel,
        depth=12,
        mlp_dim=768*2
    )

    img = torch.randn(2, in_channel, image_shape[0], image_shape[1], image_shape[2])

    v.eval()
    with torch.no_grad():
        output = v(img)
        print(output.size())



