'''Train LwF.MT model'''

import os
from models import *
from torch.autograd import Variable
from utils import saveModel, loadModel, load_cifar10_online, load_ImageNet200_online, ImgaeNet200_adjust_lr,\
    load_cifar100_online, load_MNIST_online, cifar100_adjust_lr, cifar10_adjust_lr, calc_TopK_Acc
import numpy as np
import copy
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(model, head_index, lamda, epoch, lr, train_loader, test_loader, T,
          modelPath, checkPoint, useCuda=True, adjustLR=False, earlyStop=False, tolearnce=4):

    tolerance_cnt = 0
    step = 0
    best_acc = 0

    old_model = copy.deepcopy(model)  # copy old model

    if useCuda:
        model = model.cuda()
        old_model = old_model.cuda()

    ceriation = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # train
    for epoch_index in range(1, epoch+1):

        sum_loss = 0
        old_sum_loss = 0
        new_sum_loss = 0

        model.train()
        old_model.eval()
        old_model.freeze_weight()

        if adjustLR:  # use LR adjustment
            ImgaeNet200_adjust_lr(optimizer, lr, epoch_index)

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            if useCuda:  # use GPU
                x, target = x.cuda(), target.cuda()
            test_x = Variable(x,  volatile=True)
            x, target = Variable(x), Variable(target)

            # get response from old heads
            old_outputs = []
            old_mid_out = old_model.get_middle_output(test_x)
            for head_idx in range(len(model.head_list) - 1):
                old_output = old_model.get_output(old_mid_out, head_idx)
                old_output = old_output.cpu().data.numpy()
                old_output = Variable(torch.from_numpy(old_output).cuda())
                old_outputs.append(old_output)

            # distilling loss
            old_loss = Variable(torch.zeros(1).cuda())
            new_mid_out = model.get_middle_output(x)
            for idx in range(len(old_outputs)):
                out = model.get_output(new_mid_out, idx)
                old_loss += MultiClassCrossEntropy(out, old_outputs[idx], T=T)

            # calculate new loss
            out = model.get_output(new_mid_out, head_index)
            target -= CLASS_NUM_IN_BATCH * head_index  # transform the class labels
            new_loss = ceriation(out, target)

            loss = new_loss + lamda * old_loss

            sum_loss += loss.data[0]
            old_sum_loss += old_loss.data[0]
            new_sum_loss += new_loss.data[0]

            loss.backward()
            optimizer.step()

            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.6f},'
                      ' new loss: {:.6f}, old loss: {:.6f}'.
                      format(epoch_index, batch_idx + 1, step, sum_loss/(batch_idx+1),
                             new_sum_loss/(batch_idx+1), old_sum_loss/(batch_idx+1)))

        acc = inference(model, test_loader, useCuda=True, k=5)

        #  observe new and old classes acc
        new_testLoader, new_test_classes = load_ImageNet200_online([test_root],
                                                                category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                                batchSize=batch_size, train=False)
        print("new test classes")
        new_acc = inference(model, new_testLoader, useCuda=True, k=5)
        if i != 0:
            old_testLoader, old_test_classes = load_ImageNet200_online([test_root],
                                                                    category_indexs=class_index[:i],
                                                                    batchSize=batch_size, train=False)
            print("old test classes")
            old_acc = inference(model, old_testLoader, useCuda=True, k=5)

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


def inference(model, testLoader,  useCuda=True, k=1):

    correct_cnt, sum_loss = 0, 0
    total_cnt = 0

    head_num = len(model.head_list)

    if useCuda:
        model = model.cuda()
    model.eval()

    for batch_idx, (x, target) in enumerate(testLoader):

        x = Variable(x, volatile=True)

        if useCuda:
            x = x.cuda()

        logits_list = []
        mid_out = model.get_middle_output(x)
        for head_index in range(0, head_num):

            logit = model.get_output(mid_out, head_index)
            logit = logit.cpu().data.numpy()
            logits_list.append(logit)

        total_cnt += x.data.size()[0]

        logits_array = np.hstack(logits_list)

        target_np = target.numpy()
        correct_cnt += calc_TopK_Acc(logits_array, target_np, topk=k)
        # pred_label = np.argmax(logits_array, axis=1)
        # correct_cnt += (pred_label == target.cpu().data.numpy()).sum()

    acc = (correct_cnt * 1.0 / float(total_cnt))
    print("inference acc:", acc)
    return acc


if __name__ == '__main__':

    # parameters
    TOTAL_CLASS_NUM = 200
    CLASS_NUM_IN_BATCH = 20
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    batch_size = 128
    epoch = 80
    lr = 0.1
    label_set = set()
    label_dict = {}
    lamda = 1
    temperature = 2

    # data root
    # train_root = './data/CIFAR10_png/training/'
    # test_root = './data/CIFAR10_png/testing/'
    train_root = './data/tiny-imagenet-200_1/training/'
    test_root = './data/tiny-imagenet-200_1/testing/'

    use_cuda = torch.cuda.is_available()
    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]  

    print('==> Building model..')
    # net = Simple_CNN_Ti_LWF(output_dim=CLASS_NUM_IN_BATCH)
    net = ResNet18_Ti_LWF(outputDim=CLASS_NUM_IN_BATCH)

    for i in range(0, TOTAL_CLASS_NUM, CLASS_NUM_IN_BATCH):  

        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])

        trainLoader, train_classes = load_ImageNet200_online([train_root],
                                                          category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                          train=True, batchSize=batch_size, data_pool_root=None)

        testLoader, test_classes = load_ImageNet200_online([test_root],
                                                        category_indexs=class_index[:i + CLASS_NUM_IN_BATCH],
                                                        train=False, batchSize=batch_size, shuffle=False)

        print("train classes:", train_classes)
        print("test classes:", test_classes)

        # add head
        head_index = net.add_head_layer(CLASS_NUM_IN_BATCH)

        # train and save model
        saveModelPath = "./model/imagenet200_online/resnet18_LWF_multi_head_" + str(i)

        train(model=net, head_index=head_index, lamda=lamda, epoch=epoch, lr=lr, train_loader=trainLoader,
              test_loader=testLoader, T=temperature, modelPath=saveModelPath, checkPoint=10,
              useCuda=use_cuda, adjustLR=True, earlyStop=False, tolearnce=4)

        net, best_acc, best_epoch = loadModel(saveModelPath, net)



