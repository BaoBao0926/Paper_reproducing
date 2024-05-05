# 这是写U-Net用于segmentation的文档

import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import util


class Trainer:

    def __init__(self, path, img_sava_path, model_save_path):
        self.path = path
        self.image_save_path = img_sava_path
        self.model_save_path = model_save_path
        # device cpu or gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # network
        self.net = util.UNet().to(self.device)
        # optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        # loss function, this is binary question
        self.loss_function = nn.BCELoss()
        # dataset
        self.dataloader = DataLoader(util.Eye_Datasets(self.path), batch_size=4,
                                     shuffle=True, num_workers=4)

    def train(self, stop_epoch):

        for epoch in range(stop_epoch):
            i = 1
            for inputs, labels in tqdm(self.dataloader, desc=f"Epoch {epoch}/{stop_epoch}",
                                       ascii=True, total=len(self.dataloader)):
                # get picture and labels
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # get output through the network
                out = self.net(inputs)
                # get loss
                loss = self.loss_function(out, labels)
                # backward backpropagation
                self.opt.zero_grad()    # make graident zero
                loss.backward()         # compute the graident
                self.opt.step()         # backpropagation

                # save image
                img1 = torch.stack([inputs[0], out[0], labels[0]], 0)
                img2 = torch.stack([inputs[1], out[1], labels[1]], 0)
                img3 = torch.stack([inputs[2], out[2], labels[2]], 0)
                img4 = torch.stack([inputs[3], out[3], labels[3]], 0)
                big_image = torch.cat([img1, img2, img3, img4], dim=2)
                temp = os.path.join(self.image_save_path,"result\\train1-EYE")
                save_image(big_image.cpu(), os.path.join(temp, f"Epoch{epoch}_Image{i}.png"))
                i = i+1

            if epoch % 10 == 0:
                torch.save(self.net.state_dict(), f"U-Net{epoch}")
                print("the model has been saved")


if __name__ == '__main__':
    print(torch.cuda.is_available())

    t = Trainer(path="dataset/eyes-datasets/DRIVE/training",
                img_sava_path="",
                model_save_path=r'result/train1')
    t.train(10)