'''Train LWF.MC model '''


import os
from models import *
from torch.autograd import Variable
from utils import saveModel, loadModel, load_cifar10_online, calc_TopK_Acc, load_ImageNet200_online, ImgaeNet200_adjust_lr,\
    load_cifar100_online, load_MNIST_online, cifar100_adjust_lr, cifar10_adjust_lr
import copy

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
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # train
    for epoch_index in range(1, epoch+1):

        sum_loss = 0
        model.train()
        old_model.eval()
        old_model.freeze_weight()

        if adjustLR:  # use LR adjustment
            ImgaeNet200_adjust_lr(optimizer, lr, epoch_index)

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            if useCuda:  # use GPU
                x, target = x.cuda(), target.cuda()

            x, target = Variable(x), Variable(target)
            logits = model(x)
            cls_loss = ceriation(logits, target)

            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
                dist_target = old_model(x)
                logits_dist = logits[:, :-CLASS_NUM_IN_BATCH]
                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, tempature,)
                loss = cls_loss + lamda * dist_loss
            else:
                loss = cls_loss

            sum_loss += loss.data[0]

            loss.backward()
            optimizer.step()

            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.6f}'.
                      format(epoch_index, batch_idx + 1, step, sum_loss/(batch_idx+1)))

        acc = inference(net, test_loader, useCuda=True, k=5)

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
                saveModel(model, i, best_acc, modelPath)
                return model
        else:
            if best_acc < acc:
                saveModel(model, i, acc, modelPath)
                best_acc = acc

    print("best acc:", best_acc)


def inference(model, testLoader, useCuda=True, k=1):

    total_cnt = 0
    top_correct_cnt = 0

    if useCuda:
        model = model.cuda()
    model.eval()

    for batch_idx, (x, target) in enumerate(testLoader):

        x, target = Variable(x, volatile=True), Variable(target, volatile=True)

        if useCuda:
            x, target = x.cuda(), target.cuda()

        out = model(x)

        total_cnt += x.data.size()[0]

        # _, pred_label = torch.max(out.data, 1)
        # correct_cnt += (pred_label == target.data).sum()

        # top-K acc
        out_np = out.cpu().data.numpy()
        target_np = target.cpu().data.numpy()
        top_correct = calc_TopK_Acc(out_np, target_np, topk=k)
        top_correct_cnt += top_correct

    # acc = (correct_cnt * 1.0 / float(total_cnt))
    top_acc = (top_correct_cnt * 1.0 / float(total_cnt))
    print("inference acc:", top_acc)
    return top_acc


if __name__ == '__main__':

    #  parameters
    TOTAL_CLASS_NUM = 200
    CLASS_NUM_IN_BATCH = 20
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    batch_size = 128
    epoch = 80
    lr = 0.1
    lamda = 1
    T = 2

    # data root
    # train_root = './data/CIFAR10_png/training/'
    # test_root = './data/CIFAR10_png/testing/'
    train_root = '/./data/tiny-imagenet-200_1/training/'
    test_root = './data/tiny-imagenet-200_1/testing/'

    use_cuda = torch.cuda.is_available()
    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]  

    # net = Simple_CNN_LWF(outputDim=CLASS_NUM_IN_BATCH)
    net = ResNet18_LWF(outputDim=CLASS_NUM_IN_BATCH)
    old_net = copy.deepcopy(net)

    for i in range(0, TOTAL_CLASS_NUM, CLASS_NUM_IN_BATCH):

        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])

        print('==> Building model..')

        if i != 0:
            net.change_output_dim(new_dim=i+CLASS_NUM_IN_BATCH)
        print("current net output dim:", net.get_output_dim())


        trainLoader, train_classes = load_ImageNet200_online([train_root],
                                                          category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                          train=True, batchSize=batch_size, data_pool_root=None)

        testLoader, test_classes = load_ImageNet200_online([test_root],
                                                        category_indexs=class_index[:i + CLASS_NUM_IN_BATCH],
                                                        train=False, batchSize=batch_size, shuffle=False)

        print("train classes:", train_classes)
        print("test classes:", test_classes)

        # train and save model
        saveModelPath = "./model/imagenet200_online/resnet_LWF_" + str(i + 1)

        train(model=net, old_model=old_net, epoch=epoch, lr=lr, tempature=T, lamda=lamda,train_loader=trainLoader,
              test_loader=testLoader, modelPath=saveModelPath, checkPoint=10,
              useCuda=use_cuda, adjustLR=True, earlyStop=False, tolearnce=4)

        net, best_acc, best_epoch = loadModel(saveModelPath, net)
        old_net = copy.deepcopy(net)



