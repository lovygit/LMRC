'''Train LWF.MC model with Rehearsal'''


import os
from models import *
from torch.autograd import Variable
from utils import saveModel, load_MNIST_online, load_cifar10_online, random_sample_data_pool, ImgaeNet200_adjust_lr,\
    load_cifar100_online, cifar10_adjust_lr, cifar100_adjust_lr, UnNormalize, loadModel, load_ImageNet200_online
import shutil
import copy
from LWF.MC import inference

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(model, old_model, epoch, lr, tempature, lamda, train_loader, test_loader, modelPath, checkPoint, useCuda=True,
          adjustLR=False, earlyStop=False, tolearnce=4):

    tolerance_cnt = 0
    step = 0
    best_acc = 0

    if useCuda:
        model = model.cuda()
        old_model = old_model.cuda()

    ceriation = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # train
    for epoch_index in range(1, epoch+1):

        sum_loss = 0
        sum_dist_loss = 0

        model.train()
        old_model.eval()
        old_model.freeze_weight()

        if adjustLR:  # use LR adjustment
            cifar100_adjust_lr(optimizer, lr, epoch_index)

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            if useCuda:  # use GPU
                x, target = x.cuda(), target.cuda()

            x, target = Variable(x), Variable(target)
            logits = model(x)
            cls_loss = ceriation(logits, target)

            dist_loss = Variable(torch.zeros(1).cuda())
            if i > 1:  # distill loss is not used in first class batch
                dist_target = old_model(x)
                logits_dist = logits[:, :-CLASS_NUM_IN_BATCH]
                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, tempature)
                loss = cls_loss + lamda * dist_loss
            else:
                loss = cls_loss

            sum_loss += loss.data[0]
            sum_dist_loss += dist_loss.data[0]

            loss.backward()
            optimizer.step()

            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.6f}, dist loss:{:.6f}'.
                      format(epoch_index, batch_idx + 1, step, sum_loss/(batch_idx+1), sum_dist_loss/(batch_idx+1)))

        acc = inference(net, test_loader, useCuda=True, k=1)

        # early stopping
        if earlyStop:
            if acc < best_acc:
                tolerance_cnt += 1
            else:
                best_acc = acc
                tolerance_cnt = 0
                saveModel(model, epoch_index, best_acc, modelPath)

            if tolerance_cnt >= tolearnce:
                print("early stopping training....")
                saveModel(model, epoch_index, best_acc, modelPath)
                return model
        else:
            if best_acc < acc:
                saveModel(model, epoch_index, acc, modelPath)
                best_acc = acc

    print("best acc:", best_acc)


if __name__ == '__main__':

    # parameters
    TOTAL_CLASS_NUM = 100
    CLASS_NUM_IN_BATCH = 10
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    batch_size = 128
    lr = 0.01
    epoch = 70
    sampleNum = 50
    T = 2
    lamda = 1

    # data root
    train_root = './data/CIFAR100_png/training/'
    test_root = './data/CIFAR100_png/testing/'
    data_pool_root = './data/CIFAR100_png/data_pool_2/'
    # train_root = './data/tiny-imagenet-200_1/training/'
    # test_root = './data/tiny-imagenet-200_1/testing/'
    # data_pool_root = './data/tiny-imagenet-200_1/data_pool_2/'

    use_cuda = torch.cuda.is_available()
    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]  

    # data_pool build up and clean
    if os.path.isdir(data_pool_root):
        print("data pool has been set up, start cleaning...")
        shutil.rmtree(data_pool_root)
        os.mkdir(data_pool_root)
    else:
        print("data pool has not been set up, start setting up...")
        os.mkdir(data_pool_root)

    # net = Simple_CNN_LWF(outputDim=CLASS_NUM_IN_BATCH)
    net = ResNet34_LWF(outputDim=CLASS_NUM_IN_BATCH)
    old_net = copy.deepcopy(net)

    acc_list = []

    for i in range(0, TOTAL_CLASS_NUM, CLASS_NUM_IN_BATCH):

        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])

        print('==> Building model..')

        if i != 0:
            net.change_output_dim(new_dim=i+CLASS_NUM_IN_BATCH)
        print("current net output dim:", net.get_output_dim())

        if i == 0:  
            trainLoader, train_classes = load_cifar100_online([train_root],
                                                              category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                              batchSize=batch_size, train=True, shuffle=True)
        else:  
            trainLoader, train_classes = load_cifar100_online([train_root],
                                                              category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                              batchSize=batch_size, data_pool_root=data_pool_root,
                                                              train=True, shuffle=True)

        testLoader, test_classes = load_cifar100_online([test_root],
                                                        category_indexs=class_index[0:i + CLASS_NUM_IN_BATCH],
                                                        batchSize=batch_size, train=False, shuffle=False)
        print("train classes:", train_classes)
        print("test classes:", test_classes)

        # # train and save model
        saveModelPath = "./model/cifar100_online/resnet_LWF_data_pool" + str(i+1)

        train(model=net, old_model=old_net, epoch=epoch, lr=lr, tempature=T, lamda=lamda, train_loader=trainLoader,
              test_loader=testLoader, modelPath=saveModelPath, checkPoint=10,
              useCuda=use_cuda, adjustLR=True, earlyStop=False, tolearnce=4)

        # random sampling to update data pool
        new_classes = class_index[i: i+CLASS_NUM_IN_BATCH]
        for cls_idx in new_classes:
            random_sample_data_pool(root_dir=train_root, data_pool_root=data_pool_root, class_index=cls_idx, sample_num=sampleNum)

        net, best_acc, best_epoch = loadModel(saveModelPath, net)
        old_net = copy.deepcopy(net)







