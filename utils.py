# Some helper functions for PyTorch, including:

import os
import torch
import torchvision.transforms as transforms
import torchvision
from load_data_online import CostumeImageFolder
import shutil
import numpy as np
import random
from skimage import io


def saveModel(net, epoch, best_acc, modelPath):

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': epoch,
    }
    torch.save(state, modelPath)


def loadModel(modelPath, net):

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(modelPath)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net, best_acc, start_epoch


# def load_fine_tune_model(modelPath, net, class_num, freeze_feature=False):
#
#     print('==> Resuming fine-tuning model from checkpoint..')
#     checkpoint = torch.load(modelPath)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
#
#     if freeze_feature:
#         for param in net.parameters():
#             param.requires_grad = False
#
#     net.change_output_dim(new_dim=class_num)
#
#     return net, best_acc, start_epoch
#
#
# def load_EWC_model(modelPath, net, class_num):
#
#     print('==> Resuming EWC model from checkpoint..')
#     checkpoint = torch.load(modelPath)
#     net.load_state_dict(checkpoint['net'])
#     print(checkpoint['net'].keys())
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
#
#     curr_output_dim = 0
#     for fc in net.final_fc_list:
#         curr_output_dim += fc.out_features
#
#     if class_num <= curr_output_dim:
#         return net, best_acc, start_epoch
#
#     net.add_output_dim(add_dim=class_num-curr_output_dim)
#
#     return net, best_acc, start_epoch
#
#
# def load_partial_update_model(modelPath, net, class_num):
#
#     print('==> Resuming EWC model from checkpoint..')
#     checkpoint = torch.load(modelPath)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
#
#     for param in net.parameters():
#         param.requires_grad = False
#
#     curr_output_dim = 0
#     for fc in net.final_fc_list:
#         curr_output_dim += fc.out_features
#
#     if class_num <= curr_output_dim:
#         return net, best_acc, start_epoch
#
#     net.add_output_dim(add_dim=class_num - curr_output_dim)
#
#     return net, best_acc, start_epoch


def loadCIFAR10(batchSize):
    # Data
    print('==> Preparing cifar10 data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def loadCIFAR100(batchSize):
    # Data
    print('==> Preparing cifar100 data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    return trainloader, testloader


def loadMNIST(batchSize):

    # Data
    print('==> Preparing mnist data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    return trainloader, testloader


def load_MNIST_online(roots, category_indexs, batchSize, train, data_pool_root=None, shuffle=True):

    '''
    :param roots: list of data path
    :param category_indexs: list of class index
    :param batchSize: batch size
    :param data_pool_root: data pool path
    :return: dataLoader, data_classes
    '''

    # The data are organized as a list of folders. Every folder contains the data of a class.
	# Each folder contains a sub-folder

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dirs = [root+str(i)+"/" for i in category_indexs for root in roots]  
    # if using data pool
    if data_pool_root:
        data_pool_dirs = os.listdir(data_pool_root)
        data_pool_dirs = [os.path.join(data_pool_root, dirc) for dirc in data_pool_dirs]
        dirs.extend(data_pool_dirs)
        print("dir path:", dirs)

    if train:
        transform = train_transform
    else:
        transform = test_transform

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="L")

    data_classes = list(map(int, data.classes))

    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, data_classes


def load_cifar10_online(roots, category_indexs, batchSize, train, data_pool_root=None, shuffle=True):
    '''
        :param roots: list of data path
        :param category_indexs: list of class index
        :param batchSize: batch size
        :param data_pool_root: data pool path
        :return: dataLoader, data_classes
        '''


    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots] 
    # if using data pool
    if data_pool_root:
        data_pool_dirs = os.listdir(data_pool_root)
        data_pool_dirs = [os.path.join(data_pool_root, dirc) for dirc in data_pool_dirs]
        dirs.extend(data_pool_dirs)
        print("dir path:", dirs)

    if train:
        transform = train_transform
    else:
        transform = test_transform

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")

    data_classes = list(map(int, data.classes))

    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, data_classes


def load_cifar100_online(roots, category_indexs, batchSize, train, data_pool_root=None, shuffle=True):
    '''
            :param roots: list of data path
            :param category_indexs: list of class index
            :param batchSize: batch size
            :param data_pool_root: data pool path
            :return: dataLoader, data_classes
            '''


    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots] 
    # if using data pool
    if data_pool_root:
        data_pool_dirs = os.listdir(data_pool_root)
        data_pool_dirs = [os.path.join(data_pool_root, dirc) for dirc in data_pool_dirs]
        dirs.extend(data_pool_dirs)
        print("dir path:", dirs)

    if train:
        transform = train_transform
    else:
        transform = test_transform
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")

    data_classes = list(map(int, data.classes))

    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, data_classes


def load_ImageNet200(train_root, test_root, batchSize):

    # Data
    print('==> Preparing imagenet200 data..')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.ImageFolder(root=train_root, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.ImageFolder(root=test_root,  transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    return trainloader, testloader


def load_ImageNet200_online(roots, category_indexs, batchSize, train,  data_pool_root=None, shuffle=True, img_size=224):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]  
    # if using data pool
    if data_pool_root:
        data_pool_dirs = os.listdir(data_pool_root)
        data_pool_dirs = [os.path.join(data_pool_root, dirc) for dirc in data_pool_dirs]
        dirs.extend(data_pool_dirs)
        print("dir path:", dirs)

    if train:
        transform = train_transform
    else:
        transform = test_transform
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")

    data_classes = list(map(int, data.classes))

    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, data_classes


def calc_TopK_Acc(pred, target, topk=5):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    pred_label = np.argsort(pred, axis=1)[:, ::-1]
    pred_label = pred_label[:, :topk]

    target = target.reshape((-1, 1))
    rep_target = np.tile(target, (1, min(topk, pred.shape[1])))

    bool_arr = (pred_label == rep_target)
    correct = bool_arr.sum()

    return correct


def cifar100_adjust_lr(optimizer, lr, epoch):

    # schedule lr adjust
    if epoch >= 50 and epoch < 60:
        lr *= 0.5
    if epoch >= 60 and epoch <= 70:
        lr *= (0.5 * 0.5)

    print('Learning rate: ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cifar10_adjust_lr(optimizer, lr, epoch):

    if epoch >= 35 and epoch < 45:
        lr *= 0.5
    if epoch >= 45 and epoch <= 50:
        lr *= (0.5 * 0.5)

    print('Learning rate: ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ImgaeNet200_adjust_lr(optimizer, lr, epoch):

    if epoch >= 50 and epoch < 65:
        lr *= 0.5
    if epoch >= 65 and epoch < 75:
        lr *= (0.5 * 0.5)
    if epoch >= 75 and epoch <= 80:
        lr *= (0.5 * 0.5)

    print('Learning rate: ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def random_sample_data_pool(root_dir, data_pool_root, class_index, sample_num):

    class_index = str(class_index)

    class_data_path = os.path.join(root_dir, class_index, class_index)
    images_file_names = os.listdir(class_data_path)

    random_image_file_names = np.random.choice(images_file_names, sample_num, replace=False)

    for image_name in random_image_file_names:

        image_path = os.path.join(root_dir, class_index, class_index, image_name)
        save_dir = os.path.join(data_pool_root, class_index)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_image_root = os.path.join(data_pool_root, class_index, class_index)
        if not os.path.isdir(save_image_root):
            os.mkdir(save_image_root)

        save_image_path = os.path.join(data_pool_root, class_index, class_index, image_name)

        img = io.imread(image_path)
        io.imsave(save_image_path, img)


if __name__ == '__main__':

    # trainLoader, testLoader = loadCIFAR10(128)

    # train_root = '/disk1/zhangxu_new/tiny-imagenet-200_1/training/'
    # test_root = '/disk1/zhangxu_new/tiny-imagenet-200_1/testing/'
    # data_pool_root = '/disk1/zhangxu_new/tiny-imagenet-200_1/data_pool/'
    #
    # class_indexs = [i for i in range(200)]
    # for i in range(0, 200, 20):
    #
    #     data_loader, data_classes = load_ImageNet200_online([train_root], category_indexs=class_indexs[i:i+20],
    #                                                    batchSize=128, data_pool_root=None, shuffle=False, train=True)
    #     print(data_classes)
    #
    #     for x, target in data_loader:
    #         print(x.size())
    #         # print(target)

    # clean_data(train_root)


    train_root = './data/mnist_png/training/'
    test_root = './data/mnist_png/testing/'
    data_pool_root = './data/mnist_png/data_pool/'

    random_sample_data_pool(root_dir=train_root, data_pool_root=data_pool_root,
                            class_index=1, sample_num=100)