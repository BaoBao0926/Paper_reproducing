import argparse
import copy
import os
import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from dataset import CUB_200_2011_train, CUB_200_2011_eval
import torchvision.transforms as transform
import torch
from tqdm import tqdm
from module import AlexNet

device = "gpu" if torch.cuda.is_available() else "cpu"

# 获得一个list，list中是200个dataset
def get_all_dataset(mainPath):  # mainPath是这个文件夹
    # 为了创建出一个list，这个list中有200个dataset
    train_dataset_list = []
    eval_dataset_list = []

    class_list = os.listdir(os.path.join(mainPath, "Datasets", "CUB_200_2011", "images"))
    trans = transform.Compose([
        transform.ToTensor(),
        transform.Resize((224, 224))
    ])
    for i in class_list:
        train_dataset_list.append(CUB_200_2011_train(mainPath, i, trans))
        eval_dataset_list.append(CUB_200_2011_eval(mainPath, i, trans))

    return train_dataset_list, eval_dataset_list

def train1(train_dataset1, train_dataset2, epoch, module, mainPath):

    """
     # train1 代码-虽然整体是一个一个class进行训练，但是开头，我们总不能只训练一个，所以，刚开始一起训练前两个类，写在了train1中
    Args:
        train_dataset1:
        train_dataset2:
        epoch:
        module:
        mainPath:

    Returns:

    """
    # get loader
    train_dataset = train_dataset1 + train_dataset2
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    loss_func = nn.CrossEntropyLoss()   # get loss function
    opt = torch.optim.Adam(module.parameters())     #get optimization


    for e in range(epoch):

        for batch_index, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {e}/{epoch}", ascii=True)): #, total=len(dataloader)):
        # for batch_index, (inputs, targets) in enumerate(dataloader):
            # 由于在one-hot编码的时候是class id是n，那么就创建一个n维的向量，所以在train1中，一个的大小是1，一个是2，需要统一成2
            t = targets-1
            one_hot_matrix = torch.eye(2)   # 创建一个单位矩阵作为one-hot编码的基础
            targets = one_hot_matrix[t]     # 获得one-hot编码的targets


            inputs, targets = inputs.to(device), targets.to(device)
            out = module(inputs)
            loss = loss_func(out, targets)
            # backward backpropagation
            opt.zero_grad()  # make graident zero
            loss.backward()  # compute the graident
            opt.step()  # backpropagation


        print(f"this is epoch {e} , loss is {loss}")


    # save the module path
    path = os.path.join(mainPath)+"/weight/train1/mm2.pth"
    torch.save(module.state_dict(), path)
    print("the module has been save")

    return module

def eval1(evalset1, evalset2, module, mainPath):
    module.to(device)
    module.eval()

    # get dataloader
    eval_dataset = evalset1 + evalset2
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=True, num_workers=2)

    correct_num = 0 #记录对了多少个
    total_num = len(eval_dataset)   # 总共的数量

    with torch.no_grad():

        for i, (image, targets) in enumerate(eval_dataloader):
            image, targets = image.to(device), targets.to(device)
            # 改变target，因为原来的target是class id，需要转换到0-based
            targets = targets - 1
            # one_hot_matrix = torch.eye(2)  # 创建一个单位矩阵作为one-hot编码的基础，由于第一次训练一定是2，所以直接就是2
            # targets = one_hot_matrix[t]    # 获得one-hot编码的targets
            outputs = module(image)

            # get accuracy
            index = torch.argmax(outputs, dim=1)
            for j in range(outputs.size(0)):
                correct_num = correct_num + 1 if index[j] == targets[j] else correct_num

        print(correct_num, " / ", total_num)

# train2就开始要一个一个class开始训练，想法是，在mian中传入，train2只负责训练哪一个类
def train2(train_dataset, old_model, epoch, times, main_path):
    """

    Args:
        train_dataset: 这一次训练的datset
        old_model: 上一次训练完成的模型，用作知识蒸馏的teacher
        epoch: 每一个类别训练的次数
        times: 代表现在是第几次了，训练第几个class,train2是从3开始
    Returns:

    """
    # get dataloder
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    # loss function
    loss_func = nn.CrossEntropyLoss()   # get hard_label loss function
    kd_loss_func = torch.nn.MSELoss()   # get soft_label knowledge distinctiaion function
    # freeze the parameter of teacher model
    for param in old_model.parameters():
        param.requires_grad = False

    # get new module based on the old module
    new_model = copy.deepcopy(old_model)
    new_model.fc[6] = nn.Linear(4096, times)
    # print(f"this is old model {old_model}")
    print(f"this is new model {new_model} for training times {times}")

    opt = torch.optim.Adam(new_model.parameters())  # get optimiza

    for e in range(epoch):

        for batch_index, (inputs, targets) in enumerate(
                tqdm(dataloader, desc=f"Epoch {e}/{epoch}", ascii=True)):  # , total=len(dataloader)):
            # 现在是一个class一个class的进行训练，所以他的hard label一定是一样的，都是矩阵上class_id-1的位置上然后-1
            # 所以等会直接在下面使用torch.ones(len(target)),直接得到hard label

            inputs, targets = inputs.to(device), targets.to(device)
            # get output from teacher and student model
            old_out = old_model(inputs)
            new_out = new_model(inputs)
            # get loss
            classification_loss = loss_func(new_out[:, times-1], torch.ones(len(targets)))     # hard label一定是1,大小会和batch_size一样大
            KD_loss = kd_loss_func(new_out[:, :times-1], old_out)
            total_loss = classification_loss + KD_loss
            # backward backpropagation
            opt.zero_grad()  # make graident zero
            total_loss.backward()  # compute the graident
            opt.step()  # backpropagation

        print(f"this is epoch {e} , loss is {total_loss.item()}")

    path = os.path.join(main_path)+f"/weight/train1/mm{times}.pth"
    torch.save(old_model.state_dict(), path)
    print("the module has been save")

    return new_model

def eval2(datasetList,times, model):
    """
    它需要把以前所有的eval_dataset都传入进来，然后评估一下
    datasetList:  第time次的时候，需要把前times的所有eval dataet都一起传入
    times: 第几次训练/评估
    Returns:

    """
    # get all dataset
    total_dataset = datasetList[0]
    for i in range(len(datasetList)):
        if i != 0:
            total_dataset = total_dataset + datasetList[i]
    # get dataloader
    eval_dataloader = torch.utils.data.DataLoader(total_dataset, batch_size=4, shuffle=True, num_workers=2)
    # set eval mode
    model.eval()

    correct_num = 0  # 记录对了多少个
    total_num = len(total_dataset)  # 总共的数量

    with torch.no_grad():

        for i, (image, targets) in enumerate(eval_dataloader):
            image, targets = image.to(device), targets.to(device)
            # 改变target，因为原来的target是class id，需要转换到0-based
            targets = targets - 1
            # one_hot_matrix = torch.eye(2)  # 创建一个单位矩阵作为one-hot编码的基础，由于第一次训练一定是2，所以直接就是2
            # targets = one_hot_matrix[t]    # 获得one-hot编码的targets
            outputs = model(image)

            # get accuracy
            index = torch.argmax(outputs, dim=1)
            for j in range(outputs.size(0)):
                correct_num = correct_num + 1 if index[j] == targets[j] else correct_num

        print(correct_num, " / ", total_num)

if __name__ == "__main__":
    # 测试get_all_dataset的代码
    # get_all_dataset("D:\Learning_Rescoure\extra\Project\\0.Project_Exercise\\3.Learning-without-Forgetting-using-Pytorch-main")

    # # 测试eval1的代码
    # model = AlexNet()
    # # 指定模型参数文件的路径
    # main_path = "D:\Learning_Rescoure\extra\Project\\0.Project_Exercise\\3.Learning-without-Forgetting-using-Pytorch-main"
    # model_path = os.path.join(main_path, "weight", "train1", "mm2.pth")
    # # 加载模型参数
    # model.load_state_dict(torch.load(model_path))
    # train_dataset_list, eval_dataset_list = get_all_dataset(main_path)
    # eval1(eval_dataset_list[0], eval_dataset_list[1], model, main_path)

    # # 测试train1的代码
    # model = AlexNet()
    # # print(f"main model {model}")
    # # 指定模型参数文件的路径
    # main_path = "D:\Learning_Rescoure\extra\Project\\0.Project_Exercise\\3.Learning-without-Forgetting-using-Pytorch-main"
    # model_path = os.path.join(main_path, "weight", "train1", "mm2.pth")
    # # 加载模型参数
    # model.load_state_dict(torch.load(model_path))
    # train_dataset_list, eval_dataset_list = get_all_dataset(main_path)
    #
    # mm = train2(train_dataset=train_dataset_list[2], old_model=model, epoch=10, times=3)

    #测试eval2的代码
    model = AlexNet()
    # print(f"main model {model}")
    # 指定模型参数文件的路径
    main_path = "D:\Learning_Rescoure\extra\Project\\0.Project_Exercise\\3.Learning-without-Forgetting-using-Pytorch-main"
    model_path = os.path.join(main_path, "weight", "train1", "mm2.pth")
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    train_dataset_list, eval_dataset_list = get_all_dataset(main_path)

    mm = train2(train_dataset=train_dataset_list[2], old_model=model, epoch=10, times=3,main_path=main_path)

    eval2(datasetList=eval_dataset_list[0:3], times=3, model=mm)