'''Train EWC model'''


import os
from models import *
from torch.autograd import Variable
from utils import saveModel, load_MNIST_online, load_cifar10_online, calc_TopK_Acc,\
    load_cifar100_online, cifar10_adjust_lr, cifar100_adjust_lr, load_ImageNet200_online, ImgaeNet200_adjust_lr

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def train(model, epoch, lr, train_loader, test_loader, modelPath, checkPoint=10, lamda=15, useCuda=True,
          adjustLR=False, earlyStop=False, tolearnce=4):

    tolerance_cnt = 0
    step = 0
    best_acc = 0

    if useCuda:
        model = model.cuda()

    ceriation = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # trainning
    for i in range(1, epoch+1):

        sum_loss = 0
        sum_ewc_loss = 0
        model.train()

        if adjustLR:  # use LR adjustment
            ImgaeNet200_adjust_lr(optimizer, lr, i)

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            if useCuda:
                x, target = x.cuda(), target.cuda()

            x, target = Variable(x), Variable(target)
            out = model(x)

            objective_loss = ceriation(out, target)
            ewc_loss = model.ewc_loss(lamda, cuda=use_cuda)
            loss = objective_loss + ewc_loss

            sum_loss += loss.data[0]
            sum_ewc_loss += ewc_loss.data[0]

            loss.backward()
            optimizer.step()

            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.6f}, ewc loss: {:.6f}'.
                      format(i, batch_idx + 1, step, sum_loss/(batch_idx+1), sum_ewc_loss/(batch_idx+1)))

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


def inference(model, testLoader, useCuda=True, k=1):

    # correct_cnt, sum_loss = 0, 0
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

        # calculate top-k acc
        out_np = out.cpu().data.numpy()
        target_np = target.cpu().data.numpy()
        top_correct = calc_TopK_Acc(out_np, target_np, topk=k)
        top_correct_cnt += top_correct

    # acc = (correct_cnt * 1.0 / float(total_cnt))
    top_acc = (top_correct_cnt * 1.0 / float(total_cnt))
    print("inference acc:", top_acc)
    return top_acc


if __name__ == '__main__':

    # parameters
    TOTAL_CLASS_NUM = 200
    CLASS_NUM_IN_BATCH = 20
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    batch_size = 64
    epoch = 100\

    lr = 0.1
    fisher_estimation_sample_size = 1024
    lamda = 15

    # data root
    # train_root = './data/CIFAR10_png/training/'
    # test_root = './data/CIFAR10_png/testing/'
    train_root = './data/tiny-imagenet-200_1/training/'
    test_root = './data/tiny-imagenet-200_1/testing/'

    use_cuda = torch.cuda.is_available()
    class_index = [i for i in range(0, TOTAL_CLASS_NUM)] 

    # net = Simple_CNN_EWC(outputDim=CLASS_NUM_IN_BATCH)
    net = ResNet18_EWC(outputDim=CLASS_NUM_IN_BATCH)
    # net_parallel = torch.nn.DataParallel(net, [0, 1])

    for i in range(0, TOTAL_CLASS_NUM, CLASS_NUM_IN_BATCH):

        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])

        print('==> Building model..')

        if i != 0:
            net.add_output_dim(add_dim=CLASS_NUM_IN_BATCH)
        print("current net output dim:", net.get_output_dim())


        trainLoader, train_classes = load_ImageNet200_online([train_root], category_indexs=class_index[i:i+CLASS_NUM_IN_BATCH],
                                                   batchSize=batch_size, data_pool_root=None, train=True, img_size=112,)
  
        testLoader, test_classes = load_ImageNet200_online([test_root], category_indexs=class_index[:i+CLASS_NUM_IN_BATCH],
                                                       batchSize=batch_size, train=False, shuffle=False, img_size=112,)
        print("train classes:", train_classes)
        print("test classes:", test_classes)

        # train and save model
        saveModelPath = "./model/imagenet200_online/resnet_EWC_" + str(i+1)

        train(model=net, epoch=epoch, lr=lr, train_loader=trainLoader,
              test_loader=testLoader, modelPath=saveModelPath, checkPoint=10, lamda=lamda,
              useCuda=use_cuda, adjustLR=True, earlyStop=False, tolearnce=4)

        fisher_info = net.estimate_fisher(trainLoader, fisher_estimation_sample_size, batch_size=batch_size)
        net.consolidate(fisher_info)







