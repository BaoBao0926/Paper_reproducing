
from torch.utils.data import Dataset
import os
import torchvision
import cv2
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision.transforms as transforms
import PIL.Image as Image
import torch
import torch.nn as nn
from torch.nn import functional as F

# Dataloader
class Eye_Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        trainingImgPath = os.listdir(os.path.join(self.path, "images")) # picture name
        train = os.path.join(self.path, 'images')
        self.trainingPath = [os.path.join(train, item) for item in trainingImgPath]
        print(self.trainingPath)

        labelImgPath = os.listdir(os.path.join(self.path, "1st_manual"))
        label = os.path.join(self.path, '1st_manual')
        self.lablePath = [os.path.join(label, item) for item in labelImgPath]
        print(self.lablePath)

        self.trans = transforms.ToTensor()

    def __len__(self):
        # length is the number of training pictures
        return len(self.trainingPath)

    def __trans__(self, img, size):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):

        img = cv2.imread(self.trainingPath[index]) # using openCV get img
        _, label = cv2.VideoCapture(self.lablePath[index]).read()

        #by default, openCV open picture in BGR. We need change it
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # 转成网络需要的正方形
        img = self.__trans__(img, 256)
        label = self.__trans__(label, 256)

        return self.trans(img), self.trans(label)

class VOC_Datassets(Dataset):

    def __init__(self, path):
        self.path = path

        # label
        labelImgPath = os.listdir(os.path.join(self.path, "SegmentationClass"))
        label = os.path.join(self.path, "SegmentationClass")
        self.labelPath = [os.path.join(label, item) for item in labelImgPath]

        #training image
        train = os.path.join(self.path, "JPEGImages")
        self.trainingPath = [os.path.join(label, item) for item in labelImgPath]

        for item in self.trainingPath:
            item = item[:-3] + "jpg"

        self.trans = transforms.ToTensor()

    def __len__(self):
        return len(self.trainingPath)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):

        img = cv2.imread(self.trainingPath[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.labelPath[index])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        img = self.__trans__(img, 256)
        label = self.__trans__(label, 256)

        return self.trans(img), self.trans(label)

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=C_in, out_channels=C_out,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.Down(x)

class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)

# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(3, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))


if __name__ == '__main__':
    i = 1
    dataset = Eye_Datasets("./dataset/eyes-datasets/DRIVE/training")
    for a, b in dataset:
        print(i)
        i = i+1
        print(a.shape)
        print(b.shape)
        trans_pil = transforms.ToPILImage()
        img1 = trans_pil(a)
        img1.show()
        img2 = trans_pil(b)
        img2.show()

        if i > 1:
            break
    pass