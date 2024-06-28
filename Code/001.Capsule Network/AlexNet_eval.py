import glob

import AlexNet
import utils
import torch
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

dataloaders = utils.dataloaders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _arparse(batch_size, eval_weights_path,num_class,remove):
    parser = argparse.ArgumentParser(description="Capsule Network on Alexnet")
    # trainning parameters
    parser.add_argument('--batch_size', default=batch_size, type=int)   # batch_size    50(batch size)*1000(batch number)
    parser.add_argument('--num_class', default=num_class, type=int)

    # # directory
    # parser.add_argument('--eval_save_dir', default=eval_save_dir, type=str,
    #                     help='the path directory of evaluating')
    parser.add_argument('--eval_weights_path', default=eval_weights_path, type=str, help='pretrained weight path')

    # mode
    parser.add_argument('--train', default=False, type=str, help='pretrained weight path')
    parser.add_argument('--remove',default=remove,type=bool)

    args = parser.parse_args()
    return args

def list_pth_files(folder_path):
    pth_files = []

    # 遍历目录下的所有文件
    for file_name in os.listdir(folder_path):
        # file_path = os.path.join(folder_path, file_name)
        # 判断是否为文件而非目录
        if file_name.endswith(".pth"):
                pth_files.append(file_name)
    return pth_files

def eval(args, model, test_loader, test_function):
    # get all pth file into a list
    pth_files = sorted(glob.glob(os.path.join(args.eval_weights_pth, "module_*.pth")))
    print(pth_files)
    # create txt file to record
    result_path = args.eval_weights_path + '/eval_result.txt'
    if not os.path.exists(result_path):
        f = open(result_path, 'w')
        f.close()

    best_accuracy = 0
    m = ''
    best_directory = os.path.join(args.eval_weights_path, pth_files[0])
    writer = SummaryWriter()
    for i, pth in enumerate(pth_files):
        # get pth file path
        pth_file_path = os.path.join(args.eval_weights_path, pth)
        print(f'----pth {pth_file_path}')
        # load model
        model.load_state_dict(torch.load(pth_file_path))
        # get accuracy, loss, precision, recall, F1
        accuracy, total_loss, precision, recall, F1 = test_function(model, test_loader, args)
        # draw picture in tensorboard---
        utils.draw_in_tersorboard(writer, i, accuracy, total_loss, precision, recall, F1)
        # --------------------------------------还没有修改完这里,
        print(f'\nFile {pth}:\nTotal Loss:{total_loss} The accuracy: {accuracy}\n')
        with open(result_path, 'a') as fid:
            fid.write(f'\nFile {pth}:\nTotal Loss:{total_loss} The accuracy: {accuracy}\n')
        if accuracy > best_accuracy:
            if args.remove and i != 0:
                os.remove(best_directory)
                best_directory = pth_file_path
                print(f'the file {best_directory} is removed')
            best_accuracy = accuracy
            m = f'\nFile {pth}:\nTotal Loss:{total_loss} The best accuracy: {accuracy}'
        else:
            if args.remove and i != 0:
                os.remove(pth_file_path)
                print(f'the file {pth_file_path} is removed')
    with open(result_path, 'a') as fid:     # record the best weight
        fid.write(m)
    writer.close()

def main():
    # CIFAR10: 1
    DATASET = 1

    # CapsAlexNet: 1    AlexNet: 2
    NETWORK = 1

    # CIFAR10
    if DATASET == 1:
        # CapsAlexNet
        if NETWORK == 1:
            args = _arparse(batch_size=50,
                            eval_weights_path='./Result/CIFAR10/Alexnet/CapsAlexNet/train/train_300',
                            remove=True,
                            num_class=10,
                            )
            # get datasetvddvd
            train_loader, test_loader = dataloaders.CIFAR10(batch_size=args.batch_size)
            # get model
            model = AlexNet.AlexCapsNet_CIFAR10(args.batch_size, device).to(device)
            # eval
            eval(args, model, test_loader, accu_function=utils.test_capOutput)
        # AlexNet
        if NETWORK == 2:
            args = _arparse(batch_size=1,
                            eval_weights_path='./Result/CIFAR10/Alexnet/AlexNet/train/train1',
                            remove=True,
                            num_class=10
                            )
            # get dataset
            train_loader, test_loader = dataloaders.CIFAR10(batch_size=args.batch_size)
            # get model
            model = AlexNet.AlexNet_CIFAR10().to(device)
            # eval
            eval(args, model, test_loader, accu_function=utils.test)

if __name__ == '__main__':
    main()



