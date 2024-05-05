import argparse
import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from dataset import CUB_200_2011_train, CUB_200_2011_eval
from util import get_all_dataset, train1, eval1, train2, eval2
from module import AlexNet



def main(mainPath, epoch):
    # 获取所有的dataset
    train_dataset_list, eval_dataset_list = get_all_dataset(mainPath)
    # 进行第一次训练，第一次训练要直接训练两个class
    # 得到alexnet模型
    alexnet = AlexNet()
    print("this is first two class training: ")
    mm = train1(train_dataset_list[0], train_dataset_list[1], epoch=epoch, module=alexnet, mainPath=mainPath)
    print("this is first two evaluaing: ")
    eval1(eval_dataset_list[0], eval_dataset_list[1], mm, mainPath)

    for i in range(3, len(train_dataset_list)):
        print("this is to train")
        # mm作为old model会一直被覆盖，然后传递下去，并且每一次的权重会被保存在weight/trian1文件夹下
        mm = train2(train_dataset_list[i], mm, epoch, i, mainPath)
        eval2(datasetList=eval_dataset_list,times=i, model=mm)

    print("finish all training")


if __name__ == "__main__":
    main("D:\Learning_Rescoure\extra\Project\\0.Project_Exercise\\3.Learning-without-Forgetting-using-Pytorch-main",
         epoch=10)




