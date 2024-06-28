import torch
from torch import nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint

# step 1 最刚开始的patch embedding
class PatchEmbedding(nn.Module):
    """
    这一层是最刚开始的，patch partition & linear projection
    用于把一个patch 4*4*3进行类似于词嵌入的操作，4*4*3=48维度，但是通过词嵌入可以把他的特征维度变化
    这里使用的不是线性层，使用卷积层进行，本质一样
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """
        Args:
            img_size: image_size代表图片的大小，默认为224*224那么大的，这里是int类型数据
            patch_size: path size为4，代表4*4*3的小格
            in_chans: 图片的通道数量3维
            embed_dim: 通过卷积层把一个patch的特征维度要扩展到多少，默认为96
            norm_layer: normalization层，默认为None
        Returns: [bs, num of Patch, embedding(96)]
        """
        super(PatchEmbedding, self).__init__()
        # 要做这一步的原因，我认为是因为想把图片的高和宽分开来，patch的高宽也要分开来，不过我们这里默认的图片都是正方形，所以无所谓了
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patch_resolution = [self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[0]]
        self.num_patch = self.patch_resolution[0]*self.patch_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = None

        # 用于patching embedding的卷积层
        self.proj = nn.Conv2d(in_channels=in_chans,             # 输入维度是图片的三维
                              out_channels=self.embed_dim,      # 输出维度是我们想要的词嵌入的大小，默认为96
                              kernel_size=patch_size,       # patch_size是tuple，这里的kernel_size正好可以接受这tuple，
                              stride=patch_size             # 移动距离同样也是
                              )

    def forward(self, x):
        batch_size, channel, height, weight = x.shape
        assert height == self.img_size[0] and weight == self.img_size[1], \
            f"Input image size ({height}*{weight}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)    # [batch_size, 3, H, W]->[batch_size, embedding(96), H/4, W/4]
        x = x.flatten(2)    # [bs, embedding(96), H/4, W/4] -> [bs, embedding, H*W/16]
        x = x.transpose(1, 2) # [bs, H*W/16, embedding(96)]
        x = self.norm(x) if self.norm is not None else x
        return x    # [bs, num of Patch, embedding(96)]

# step 2 把window attention写了，作为一个标准Swim transformer block的一部分
class WindowAttention(nn.Module):
    """
    进行window Attention，一个window默认有7*7个patch，我们想让这7*7个patch进行一次attention
    这个是通用的，S-WA和WA都用这一个方法，在传入的时候决定要不要传入Mask从而进行区分
    所以这是第二个要写的
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: 为词嵌入的维度大小
            window_size(tuple[int]): 一个window的大小，要求是一个， [height, weight]
            num_heads: 注意力的头的数量
            qkv_bias(bool): 在变换qkv的时候是否需要bias，在基本的transofrmer中，是不需要的
            qk_scale(float): 在attention的时候需要除以维度的根号来保证数值稳定，但是这里多了一个选项，就是这个数值可以自己填入一个
            attn_drop(float): 对于attention完了之后是否需要Dropout，这个是比率
            proj_drop(float): 对于output的Dropout
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale or self.head_dim**-0.5  # 神奇的语法，如果qk_scale不是None，值为前者，否则为后者

        # 接下来是做 相对位置编码 有点难理解的，这块应该主要设计到相对位置编码应该怎么做，先不管这里是怎么写的了
        # 可以对着下面这个视频反复观看思考下，
        # https://www.bilibili.com/video/BV1zT4y197Fe?vd_source=80b346be9e1c1a93109688bf064e5be1
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 注册时候，是不可训练的参数，index是不用训练的，但是index索引过去的是可训练的参数
        self.register_buffer("relative_position_index", relative_position_index)

        # 进行attention的qkv的转换，这里还是，输出的维度理论上transformer应该用三个单独的linear层，
        # 但是这里和ViT保持一致，一起输出，然后在进行分割成qkv，可以保持矩阵计算
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # timm.models.layers.trunc_normal_ 这个可以生成一个被截断的正太分布的初始化参数
        # 相当于给self.relative_position_bias_table初始化了值
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)   # attention score的softmax一下的那个东西

    def forward(self, x, mask=None):
        """
        Args:
            x:  大小 [num_window*B, N, C] 由于这里的是在一个window内部进行计算，
        所以这里选择把window和batch弄到了一个维度里面，N代表一个window里面有多少个patch(49)，C代表词嵌入的维度
            mask: mask大小为[num_windows, Wh*Ww, Wh*Ww] or None
        """
        B_, N, C = x.shape
        # [bs, num of Patch, embedding(96)]
        qkv = self.qkv(x)   # [num_window*B, N, C]->[num_window*B, N, C*3]
        # [B*num_window, N, 3, 头的数量，单头的词维度], 3代表把qkv三个矩阵拼接在一起了
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        # [3, B*num_window, 头的数量，N， 单头的维度]，变成这样子了之后，就是很标准的attention的样子了
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    # 由于第一个维度是3，所以可以一个一个取出来
        q = q * self.qk_scale   # 进行缩放，对attention缩放还是对q或者k缩放本质都一样，
        attn = (q @ k.transpose(-2, -1))    # 这里的@代表了矩阵的乘法

        # 从相对位置编码的index表中加到attention中，这里就暂时不管了
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # 进行attention score的dropout
        attn = self.attn_drop(attn)

        # 把attention score和value相乘，最后把形状转变成原来的样子
        # 这里很巧妙，我一直觉得这种转化按都是很巧妙地，我不管你们究竟是谁和谁合并，我就要这些tensor给我变成我想要的样子
        # 如果是我写，我可能会用einops去可视化的写出来，也比较好理解每一步代码的情况
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 这一步相当于 把子空间(也就是每一个单头)通过一个线性层转变一下
        x = self.proj(x)
        # 做线形成输出的dropout，这里写一下我对dropout的感觉就是随时随地都可以dropout
        x = self.proj_drop(x)
        return x

# step 3 FFN，也是标准Swim transformer block的一部分
class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layers=nn.GELU, drop=0.):
        """
        我觉得这块的代码写的很有脑子，我从来没想过还能这么写代码，太厉害了
        Args:
            in_features: 输入的词嵌入的维度
            hidden_features:    中间层的输入的维度
            out_features: 输出的词嵌入的维度
            act_layers: 激活函数，默认为nn.GELU
            drop: drop的系数
        """
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layers()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 在构造Swim transformer block的时候，需要用到两个方法
# 第一个是 划分window 最后输出 [num_windows*B, window_size, window_size, C]
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
# 第二个是 进行window reverse，把划分成一个一个window的恢复过来 [B, H, W, C]
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# step 4, 构建一个标准的完整swim transformer block
class SwinTransformerBlock(nn.Module):
    """
    上面的class WindowAttention是既可以作为标准的window attention，又可以作为shift attention的
    一个标准的Swin TR block是由SWA和S-SWA共同构成的，所以这里想要做的是构建一个连接在一起的东西出来
    dim (int):词嵌入的大小
        input_resolution (tuple[int]): 输入的resolution，代表有 几*几 个patch，后面进行patch merge的时候resolution会越来越小的
        num_heads (int):多头的数量。
        window_size (int):窗口的大小，多少个patch*patch
        shift_size (int): SW-MSA的移位大小，就是如果一个window为7*7的patch，从视频里面看，他会向下移动3个patch，那么移动3个是我们人为指定的，我原来以为是除以2得到的
        mlp_ratio (float): mlp隐藏层的维度大小与词嵌入维度大小的比率
        qkv_bias (bool，可选):如果为True，则为query, key, value添加可学习的偏差。默认为true，用于传入WindowAttention
        qk_scale (float |无，可选):覆盖head_dim ** -0.5的默认qk比例。用于传入WindowAttention
        drop (float，可选):dropout, 默认为0.0，这里的drop应该为SA中的proj_dropout
        attn_drop (float，可选):用于attention的dropout。默认值为0，用于传入WindowAttention
        drop_path (float，可选):随机深度速率。默认值:0.0
        act_layer (nn。模块(可选):激活层。默认为 nn.GELU
        norm_layer (nn。模块(可选):规范化层。默认值:神经网络。LayerNorm
        fused_window_process (bool，可选):如果为True，则使用一个内核来融合窗口移位和窗口分区以加速，类似于反向部分。默认值:假
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):

        super(SwinTransformerBlock, self).__init__()
        self.dim = dim  # 输入的维度
        self.input_resolution = input_resolution    # 输入的resolution,[height patch num, width patch num]
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 当window size大于了input resolution，那么window就不可以被分割，也就不可以进行shift
        # 比如现在的图片是7*7 patch，而我们规定一个window是9*9（当然正常不会出现这种情况），这时候就不可以做shift分割操作了
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # 这里表示，我们进行shift的大小一定要小于window的大小，做一步确保
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),  # 这里要求输入的是一个tuple
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,  # 如果想要dim的根号，那就在最外面传入None就好了
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # 这里dropPath是把某一个残差的block给去掉，第一次见 感觉很震惊，参考如下文章
        # https://blog.csdn.net/beginner1207/article/details/138034012?ops_request_misc=&request_id=&biz_id=102&utm_term=Droppath&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-138034012.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187
        # 这里可以更看出nn.Identity()的作用，如果没有DropPath的情况下，可以使用nn，Identity代替，这样层数是一样的，虽然实际没有什么用处
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_didden_dim = int(dim * mlp_ratio)   # 中间隐藏层的维度大小是通过一个ratio算出来的
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_didden_dim,
                       act_layers=act_layer,    # 这个class就传入了一个nn.GELU
                       drop=drop
                       )

        # 这里，如果要做的是shifted attention，那么就需要生成对并的掩码，生成掩码的部分先跳过，直接粘贴过来
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # 这个方法也不管了，直接粘贴过来
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:   # 如果shift_size没有大于0，也就是不需要进行shift attention，就是标准的，所以不需要mask
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        # 这个是什么多了一个内核用来加速什么什么的
        self.fused_window_process = fused_window_process;

    def forward(self, x):
        H, W = self.input_resolution    # input resolution: [height patch num, weight patch num]
        B, L, C = x.shape       # x:[batch_size, number_patch, embedding dim]
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)   # x:[bs, patch num, embedding dim]
        # 这里相当于分成了块块，这样就可以进行shift了
        x = x.view(B, H, W, C)  # x:[bs, height patch num, width patch num, embedding dim]

        # 进行shift的判断和位移，这一步是对img进行的，下面有一个反shift，就是计算完成了之后的
        if self.shift_size > 0:
            if not self.fused_window_process:
                # torch.roll可以实现cyclic shift的操作，有时候我都不知道是先有了这个方法再有了swim TRS，还是反过来
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)   # [nW*B, window_size, window_size, C]
            else:
                # 这里需要kernels文件夹里面的东西，暂时不管
                # x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
                x_windows = 1   # 这一行不会运行到的
        else:   # 不用shift的
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, window_size*window_size, C]

        # 为了进行window attention，把所有的window都叠起来，这样就可以进行
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)

        # 进行window attention，是否shifted的都在一起了，通过self.shift_size的判断，高手写代码都喜欢写到一起
        # 这里的self.attn_mask是__init__里面最后进行的，使用register注册出来的，所以找不到那里搞的
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows, 明明是把patch拆开了 恢复原图大小，这里要叫merge，其实有点奇怪
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # 做完attention之后，reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                # [Bs, height patch num, weidth patch num, embedding dim]
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                # x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
                pass
        else:   # 没有shift的时候，标准Window attention,不需要reverse shift
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x

        x = x.view(B, H * W, C)
        # 我擦，这个drop_path尽然这么用，但是我不太懂，每一个dropath都是单独的一个对象，也是有一定概率把这个path去掉
        # 就算概率低，会不会出现一种情况，都被drop掉了？感觉这个是dropath内部实现的问题
        x = shortcut + self.drop_path(x)

        # 左后的前向 FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# step 5, 一个标准的 swim transformer block完了之后，需要来一个patch merging，减少resolution，并增加通道数
class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        """
        Args:
        input_resolution (tuple[int]): Resolution of input feature.,这是一个tuple注意
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        self.input_resolution = input_resolution
        self.dim = dim
        # 合并之后，相当于把四个embedding接到了一起，所以是4*dim，通过线性层输出成2*dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x):
        """
        Args:
            x:  [batch_size, height_patch_num * width_patch_num, embedding dim]
        Returns:
                [batch_size, height_patch_bum*width_path_num / 4, embedding dim*2]
        """
        H, W = self.input_resolution        # H:height_patch_num  W:width_patch_num
        B, L, C = x.shape   # [batch_size, height_patch_num * width_patch_num, embedding]
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # 他们这里的merging我感觉挺奇怪的，并不是相邻的四个，而是隔一个
        # 代码就是用切片slice的想法进行的，第一个维度是batch size，最后一个维度是embedding，所以全要，其他的就是各一个找一个
        x0 = x[:, 0::2, 0::2, :]    # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]    # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]    # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]    # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], dim=-1)     # B H/2 W/2 4*C
        x = x.view(B, -1, 4*C) # [B, H*W/4, 4*C]

        x = self.norm(x)
        x = self.reduction(x)   # 线性层降维度的

        return x

# step 6, 标准的一层为 n个Swim Transformer Block+1个patch merging
class BasicLayer(nn.Module):
    """
    Args:
        dim: 一个basic layer中的词嵌入维度是不变化的，除了最后的patch merging，不过那块也是输出了
        input_resolution: 分成path的resolution是什么，在官方代码中，每一个这个的使用都是tuple
        depth: int 要在patch merging的前面放多少个标准的Swim transformer block
        num_heads:  attention的时候分几个头
        window_size: window的大小，一个window是 几*几 个patch
        mlp_ratio:  mlp隐藏层里面的倍率是多少
        qkv_bias:   qkv计算的时候，是否需要使用bias，默认为true
        qk_scale:   qkv除以的那个数是否要指定，默认为none，如果为none就是dim的根号
        drop:       drop的比率
        attn_drop:  attention输出之后的drop的比率
        drop_path:  drop path的比率
        norm_layer: normalization layer为什么，默认为nn.layer
        downsample:  layer的最后的下采样层， 默认为none, type为nn.Module, 可能应该是把patchMerging给传进来了
        use_checkpoint: 是否使用checkpointing去保存memory，默认为none
        fused_window_process:   如果这个为true，会使用kernel file里面的文件加速window shift & window partition
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super(BasicLayer, self).__init__()
        self.dim = dim  # 词嵌入的维度
        self.input_resolution = input_resolution    # tuple [height, width]
        self.depth = depth   # 深度
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                # 怪不得depth非要是偶数，使用这个实现的，而且这个shift的格数还真是除以2得到的，虽然也可以认为指定
                shift_size=0 if (i % 2 == 0) else window_size//2,
                mlp_ratio=mlp_ratio,
                qk_scale=qk_scale,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                fused_window_process=fused_window_process
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                # torch.utils.checkpoint  一种保存方法，时间换空间，正向传递不保存中间值，反向传递在重新计算
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# step 7, 构建最后的最后的 吧basic layer连接起来的 swim transformer
class SwinTransformer(nn.Module):
        """
        Args:
            img_size:  图片大小，int类型，表述输入的图片有多大，默认值为224
            patch_size: 一个patch的大小是读诵好，int类型，标准是4*4个像素
            in_chans: 输入图片的channel数量
            num_classes: 刚开始是做分类任务的，所以需要知道有多少个class
            embed_dim: 词嵌入的维度大小
            depths: ！这里是一个list，list有几个元素相当于有几个basiclayer
            num_heads:  多头有几个头,也需要传入一个list，每一个阶段需要多少个头是不一样的
            window_size:    一个window几个patch
            mlp_ratio:  中间隐藏层的ratio
            qkv_bias:   qkv矩阵生成时候的偏执
            qk_scale:   qk的缩放是我们自己指定，如果是None，那么就是dim的根号
            drop_rate:  dropout rate
            attn_drop_rate: attention dropout rate
            drop_path_rate: dropath rate
            norm_layer: nn.LayerNorm.默认都是这个
            ape:    是否要使用绝对位置编码
            patch_norm: 默认是fasle，是否使用normalization在patch embedding之后
            use_checkpoint: 是否使用checkpoint去保存
            fused_window_process:是否使用高级的kernels去加速保存
            **kwargs:
        """
        def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
            super().__init__()

            self.num_classes = num_classes
            self.num_layers = len(depths)
            self.embed_dim = embed_dim
            self.ape = ape      # 这个是绝对位置编码
            self.patch_norm = patch_norm
            # embedding_dim ** stage的数量-1 *2,不知道为什么这个是最后的输出的样子
            self.num_features = int(embed_dim*2 ** (self.num_layers-1))
            self.mlp_ratio = mlp_ratio

            self.patch_embed = PatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None
            )

            # 从上面的patch embedding中找到 有多少个patch，patch的resolution是多少
            num_patches = self.patch_embed.num_patch
            patches_resolution = self.patch_embed.patch_resolution
            self.patches_resolution = patches_resolution

            # 是否使用绝对位置编码
            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
                trunc_normal_(self.absolute_pos_embed, std=.02)

            self.pos_drop = nn.Dropout(drop_rate)

            # 对于不同的depth，他的drop patch rate是不同的，我前面写的关于dropath的疑惑，这样至少保证了整个模型有一个标准的block
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

            # 构造整体的模型
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(
                    dim=int(embed_dim*2 ** i_layer),
                    input_resolution=(patches_resolution[0]//(2**i_layer),  # 还真是，每一个阶段的resolution是随着阶段数量
                                      patches_resolution[1]//(2**i_layer)),  # 不断除以2，所以用这样的方式可以直接得到
                    depth=depths[i_layer],      # 不要搞混了，这里是进行basic layer的构造，也就是一个阶段一个阶段的构造
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    # 这个发黄没有关系，默认为0，但是里面的操作处理是对list进行处理的
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    # drop_path=0.1,
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers-1) else None,
                    use_checkpoint=use_checkpoint,
                    fused_window_process=fused_window_process
                                   )
                self.layers.append(layer)

            self.norm = norm_layer(self.num_features)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes>0 else nn.Identity()

            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        @torch.jit.ignore
        def no_weight_decay(self):
            return {'absolute_pos_embed'}

        @torch.jit.ignore
        def no_weight_decay_keywords(self):
            return {'relative_position_bias_table'}

        def forward_features(self, x):
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

            for layer in self.layers:
                x = layer(x)


            x = self.norm(x)  # B L C
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            return x

        def forward(self, x):
            x = self.forward_features(x)
            x = self.head(x)
            return x



if __name__ == "__main__":

    # block = SwinTransformerBlock(
    #         dim=96,
    #         input_resolution=(20,20),
    #         num_heads=8,
    #         window_size=7,
    #     )



    # 就假设全部使用默认的
    mm = SwinTransformer()
    img = torch.randn((10, 3, 224, 224))
    output = mm(img)
    print(output.size())
