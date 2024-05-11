import math
import shutil
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

import swin_transformer_muyi

device = "cuda" if torch.cuda.is_available() else "cpu"

def FLOWER102(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.transforms.RandomRotation(0.5),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.Flowers102(root='D:\Learning_Rescoure\extra\Dataset', split='test', download=True,
                                                    transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    val_dataset = torchvision.datasets.Flowers102(root='D:\Learning_Rescoure\extra\Dataset', split='val', download=True,
                                                  transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return train_loader, val_loader

def FOOD101(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.Food101(root='.\Dataset\Food101', split='train', download=True,
                                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.Food101(root='.\Dataset\Food101', split='test', download=True,
                                                transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    return train_loader, test_loader

def MNIST(batch_size):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.MNIST(root='D:\Learning_Rescoure\extra\Dataset\MNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.MNIST(root='D:\Learning_Rescoure\extra\Dataset\MNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader


def train(model, train_loader, eval_loader, stop_epoch, lr=0.0001):
    # get gpu,or cpu
    optimizer = Adam(model.parameters(), lr=lr)  # optimizer
    loss_function = nn.CrossEntropyLoss().to(device)
    mm.to(device)

    for epoch in range(stop_epoch):

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{stop_epoch}",
                                   ascii=True, total=len(train_loader)):
            # get picture and labels
            inputs, labels = inputs.to(device), labels.to(device)
            # get output through the network
            out = model(inputs)
            # get loss
            loss = loss_function(out, labels)
            # backward backpropagation
            optimizer.zero_grad()    # make graident zero
            loss.backward()         # compute the graident
            optimizer.step()         # backpropagation


        if (epoch+1) % 100 == 0:
            torch.save(model.net.state_dict(), f"U-Net{epoch}")
            print("the model has been saved")

        # 评估正确率的代码
        eval(model, eval_loader, epoch)

def eval(model, eval_loader, epoch):
    # get gpu,or cpu
    print('-----------------------eval-----------------------')
    model = model.to(device)

    total_image_number = len(eval_loader.dataset)
    loss_function = nn.CrossEntropyLoss().to(device)
    model.eval()

    with torch.no_grad():

        correct_num = 0

        for i, (image, targets) in enumerate(eval_loader):
            image, targets = image.to(device), targets.to(device)

            outputs = model(image)   # predict output

            # get accuracy
            index = torch.argmax(outputs, dim=1)
            for j in range(outputs.size(0)):
                correct_num = correct_num + 1 if index[j] == targets[j] else correct_num


    accuracy = correct_num/total_image_number   # calculate accuracy

    print(f"epoch: {epoch}, accuracy: {accuracy}")

    with open("result.txt", "w") as file:
        # 使用write()方法写入内容
        file.write(f"epoch: {epoch}, accuracy: {accuracy} \n")



if __name__ == "__main__":
    train_loader, eval_loader = FOOD101(4)
    # train_loader, eval_loader = MNIST(16)
    mm = swin_transformer_muyi.SwinTransformer(in_chans=3)
    train(mm, train_loader, eval_loader, 100, lr=0.0001)