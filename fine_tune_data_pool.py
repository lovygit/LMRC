'''Train fine tune model with Rehearsal'''


import os
from models import *
from torch.autograd import Variable
from utils import saveModel, load_cifar100_online, load_cifar10_online,  random_sample_data_pool,\
    load_MNIST_online, cifar100_adjust_lr, cifar10_adjust_lr, load_ImageNet200_online, ImgaeNet200_adjust_lr
from torchvision import transforms
import random
import shutil
from fine_tuning import inference


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_class_weight(train_loader, train_classes):

    y_collector = [0] * len(train_classes)

    for batch_idx, (x, target) in enumerate(train_loader):
        target = target.numpy()
        for y in target:
            y_collector[y] += 1

    tmp = 1
    for num in y_collector:
        tmp *= num

    for i in range(len(y_collector)):
        y_collector[i] = tmp / y_collector[i]

    return y_collector


def train(model, epoch, lr, train_loader, test_loader, modelPath, checkPoint, useCuda=True,
          adjustLR=False, earlyStop=False, tolearnce=4):

    tolerance_cnt = 0
    step = 0
    best_acc = 0

    if useCuda:
        model = model.cuda()

    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # trainning
    for i in range(1, epoch+1):

        sum_loss = 0
        model.train()

        # class_weight = get_class_weight(trainLoader, train_classes)
        # class_weight = torch.FloatTensor(class_weight).cuda()

        if adjustLR:  # use LR adjustment
            ImgaeNet200_adjust_lr(optimizer, lr, i)

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            if useCuda:  # use GPU
                x, target = x.cuda(), target.cuda()

            x, target = Variable(x), Variable(target)
            out = model(x)

            loss = nn.CrossEntropyLoss()(out, target)
            sum_loss += loss.data[0]

            loss.backward()
            optimizer.step()

            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.6f}'.
                      format(i, batch_idx + 1, step, sum_loss/(batch_idx+1)))

        acc = inference(net, test_loader, useCuda=True, k=5)

        # early stopping
        if earlyStop:
            if acc < best_acc:
                tolerance_cnt += 1
            else:
                best_acc = acc
                tolerance_cnt = 0
                saveModel(model, i, best_acc, modelPath)

            if tolerance_cnt >= tolearnce:
                print("early stopping training....")
                saveModel(model, i, best_acc, modelPath)
                return model
        else:
            if best_acc < acc:
                saveModel(model, i, acc, modelPath)
                best_acc = acc

    print("best acc:", best_acc)


if __name__ == '__main__':

    # parameters
    TOTAL_CLASS_NUM = 200
    CLASS_NUM_IN_BATCH = 20
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    batch_size = 128
    epoch = 80
    lr = 0.1
    sampleNum = 20
    data_pool = []

    # data root
    # train_root = './data/mnist_png/training/'
    # test_root = './data/mnist_png/testing/'
    # data_pool_root = './data/mnist_png/data_pool_2/'

    train_root = './data/tiny-imagenet-200_1/training/'
    test_root = './data/tiny-imagenet-200_1/testing/'
    data_pool_root = './data/tiny-imagenet-200_1/data_pool_2/'

    # data_pool build up and clean
    if os.path.isdir(data_pool_root):
        print("data pool has been set up, start cleaning...")
        shutil.rmtree(data_pool_root)
        os.mkdir(data_pool_root)
    else:
        print("data pool has not been set up, start setting up...")
        os.mkdir(data_pool_root)

    use_cuda = torch.cuda.is_available()
    class_index = [i for i in range(0, TOTAL_CLASS_NUM)] 

    # net = Simple_CNN_fine_tune(outputDim=CLASS_NUM_IN_BATCH)
    net = ResNet18_fine_tune(outputDim=CLASS_NUM_IN_BATCH)

    for i in range(0, TOTAL_CLASS_NUM, CLASS_NUM_IN_BATCH):

        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])

        print('==> Building model..')

        if i != 0:
            net.change_output_dim(new_dim=i+CLASS_NUM_IN_BATCH)
        print("current net output dim:", net.get_output_dim())

 
        if i == 0:  
            trainLoader, train_classes = load_ImageNet200_online([train_root], category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                           batchSize=batch_size, train=True)
        else:  
            trainLoader, train_classes = load_ImageNet200_online([train_root], category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                           batchSize=batch_size, data_pool_root=data_pool_root, train=True)

        testLoader, test_classes = load_ImageNet200_online([test_root], category_indexs=class_index[0:i + CLASS_NUM_IN_BATCH],
                                                     batchSize=batch_size, train=False, shuffle=False)
        print("train classes:", train_classes)
        print("test classes:", test_classes)

        # train and save model
        saveModelPath = "./model/imagenet200_online/resnet18_fine_tune_data_pool" + str(i + 1)

        train(model=net, epoch=epoch, lr=lr, train_loader=trainLoader,
              test_loader=testLoader, modelPath=saveModelPath, checkPoint=10,
              useCuda=use_cuda, adjustLR=True, earlyStop=False, tolearnce=4)

        # random sampling to updated data pool
        new_classes = class_index[i: i + CLASS_NUM_IN_BATCH]
        for cls_idx in new_classes:
            random_sample_data_pool(root_dir=train_root, data_pool_root=data_pool_root, class_index=cls_idx,
                                    sample_num=sampleNum)





