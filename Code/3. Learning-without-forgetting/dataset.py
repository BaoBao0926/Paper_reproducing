import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transform
from PIL import Image


# 每一个dataset都只存一个class的东西
class CUB_200_2011_train(Dataset):

    # mainPath为"Datasets/CUB_200_2011", classPath是image里面具体一个class 比如”001.Black“，所以中间差了一个image需要加上
    # 再main.py中，应该会用for循环直接弄出来
    def __init__(self, mainPath, classPath, trans):
        self.mainPath = mainPath  # "D:\Learning_Rescoure\extra\Project\\0.Project_Exercise\\3.Learning-without-Forgetting-using-Pytorch-main"
        self.datasetPath = os.path.join(mainPath, "Datasets", "CUB_200_2011")
        self.classPath = classPath
        self.trans = trans

        # 找到这个class的id是什么
        with open(os.path.join(self.datasetPath, "classes.txt")) as f:
            for line in f:
                class_id, class_name = line.split()
                if self.classPath == class_name:
                    self.class_id = int(class_id)

        # 根据class的id，找到图片的id
        self.image_id = []
        with open(os.path.join(self.datasetPath, "image_class_labels.txt")) as f:
            for line in f:
                image_id, class_id = line.split()
                if int(class_id) == self.class_id:
                    self.image_id.append(image_id)

        # 通过图片的id，找到图片的地址，顺便建立一个字典
        self.id2path = {}           # 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
        with open(os.path.join(self.datasetPath, "images.txt")) as f:
            for line in f:
                image_id, image_path = line.split()
                if image_id in self.image_id:
                    self.id2path[image_id] = image_path

        # 建立training dataset
        self.training_path = []
        self.evaluating_path = []
        with open(os.path.join(self.datasetPath, "train_test_split.txt")) as f:
            for line in f:
                image_id, is_train = line.split()
                if image_id in self.image_id:  # 是这个class的
                    if int(is_train):  # 是训练样本
                        self.training_path.append(self.id2path[image_id])
                    else:
                        self.evaluating_path.append(self.id2path[image_id])

    def __len__(self):
        return len(self.training_path)  # 返回有多少个训练样本

    def __getitem__(self, index):
        training_image_path = self.training_path[index]  # 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
        image_path = os.path.join(self.datasetPath, "images", training_image_path)
        image = Image.open(image_path)
        image = self.trans(image)

        # 需要让class_id从数字变成one-hot编码
        c_id = create_one_hot(self.class_id)

        return image, self.class_id  # -1是因为编码成one-hot，提前-1方便编码


# 每一个dataset都只存一个class的东西
class CUB_200_2011_eval(Dataset):

    # mainPath为"Datasets/CUB_200_2011", classPath是image里面具体一个class 比如”001.Black“，所以中间差了一个image需要加上
    # 再main.py中，应该会用for循环直接弄出来
    def __init__(self, mainPath, classPath, trans):
        self.mainPath = mainPath
        self.datasetPath = os.path.join(mainPath, "Datasets", "CUB_200_2011")
        self.classPath = classPath
        self.trans = trans

        # 找到这个class的id是什么
        with open(os.path.join(self.datasetPath, "classes.txt")) as f:
            for line in f:
                class_id, class_name = line.split()
                if self.classPath == class_name:
                    self.class_id = int(class_id)

        # 根据class的id，找到图片的id
        self.image_id = []
        with open(os.path.join(self.datasetPath, "image_class_labels.txt")) as f:
            for line in f:
                image_id, class_id = line.split()
                if int(class_id) == self.class_id:
                    self.image_id.append(image_id)

        # 通过图片的id，找到图片的地址，顺便建立一个字典
        self.id2path = {}
        with open(os.path.join(self.datasetPath, "images.txt")) as f:
            for line in f:
                image_id, image_path = line.split()
                if image_id in self.image_id:
                    self.id2path[image_id] = image_path

        # 建立evaluating dataset
        self.evaluating_path = []
        with open(os.path.join(self.datasetPath, "train_test_split.txt")) as f:
            for line in f:
                image_id, is_train = line.split()
                if image_id in self.image_id:  # 是这个class的
                    if not int(is_train):  # 是训练样本
                        self.evaluating_path.append(self.id2path[image_id])

    def __len__(self):
        return len(self.evaluating_path)  # 返回有多少个测试样本

    def __getitem__(self, index):
        evaluating_image_path = self.evaluating_path[index]  # 这是class文件夹的地址
        image_path = os.path.join(self.datasetPath, "images", evaluating_image_path)
        image = Image.open(image_path)
        image = self.trans(image)

        # 需要让class_id从数字变成one-hot编码
        # c_id = create_one_hot(self.class_id)

        return image, self.class_id  # -1是因为编码成one-hot，提前-1方便编码

# 把lable编码成one-hot编码
def create_one_hot(n):
    if n < 1:
        raise ValueError("n must be greater than 1 for one-hot encoding.")
    one_hot = np.zeros(int(n))

    # 将第 (n-1) 行的元素设为1
    one_hot[n - 1] = 1

    return one_hot

if __name__ == "__main__":
    trans = transform.Compose([
        transform.ToTensor(),
        transform.Resize((224, 224))
    ])

    cub = CUB_200_2011_train(
        mainPath="D:\Learning_Rescoure\extra\Project\\0.Project_Exercise\\3.Learning-without-Forgetting-using-Pytorch-main\Datasets\CUB_200_2011",
        # classPath="001.Black_footed_Albatross",
        classPath="013.Bobolink",
        trans=trans)

    print(len(cub))

    for image, class_id in cub:
        print(class_id)
        # print(image)
