'''Train LMRC model with Rehearsal.'''

import os
from models import *
from torch.autograd import Variable
from utils import saveModel, loadModel, load_cifar10_online, load_cifar100_online, random_sample_data_pool,\
    load_MNIST_online, cifar10_adjust_lr, cifar100_adjust_lr, UnNormalize, calc_TopK_Acc, load_ImageNet200_online, ImgaeNet200_adjust_lr
from label_mapping import labels2Vec, label_greedy_mapping_online, label_vecs_quelity
import numpy as np
import copy
import shutil
from sklearn.metrics.pairwise import cosine_similarity


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class CosineLoss(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, y, sample_weight=None):

        n = x.size()[0]
        a = torch.bmm(x.view(n, 1, self.dim), y.view(n, self.dim, 1))
        a = a.view(n,)
        norm1 = torch.norm(x, p=2, dim=1)
        norm2 = torch.norm(y, p=2, dim=1)
        cos_loss = a / (norm1 * norm2)
        if sample_weight is not None:
            cos_loss = cos_loss * sample_weight
        mean_cos_loss = torch.mean(cos_loss)
        return -mean_cos_loss


def label_allotter(label_set, train_y, label_dict, dim, threshold=0.15):

    train_y_set = set(train_y)

    new_label = train_y_set - label_set   # take diff
    for label in new_label:
        label_set.add(label)
        label_vec = label_greedy_mapping_online(label_dict, dim, threshold=threshold, normal=True)
        label_dict[label] = label_vec

    return label_set, label_dict, new_label


def get_sample_weight(train_loader,):

    from collections import Counter
    counter = Counter()
    y_collector = []

    for batch_idx, (x, target) in enumerate(train_loader):
        x = x.numpy()
        target = target.numpy()
        for y in target:
            counter[y] += 1
            y_collector.append(y)

    n_samples = len(y_collector)
    n_classes = len(counter.keys())

    for key in counter.keys():
        counter[key] = float(n_samples / (n_classes * counter[key]))

    return counter


def train(model, cur_class, head_index, lamda, epoch, lr, output_dim, train_loader, test_loader, label_dict,
          modelPath, checkPoint, useCuda=True, adjustLR=False, earlyStop=False, tolearnce=4):

    tolerance_cnt = 0
    step = 0
    best_acc, best_new_acc, best_old_acc = 0, 0, 0

    counter = get_sample_weight(train_loader)
    print("sample weight:", counter)

    old_model = copy.deepcopy(model)  # copy the old model

    if useCuda:
        model = model.cuda()
        old_model = old_model.cuda()

    ceriation = CosineLoss(output_dim)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=lr)

    # train
    for epoch_index in range(1, epoch+1):

        sum_loss = 0
        old_sum_loss = 0
        new_sum_loss = 0

        model.train()
        old_model.eval()
        # old_model.freeze_weight()

        if adjustLR:  # use LR adjustment
            cifar10_adjust_lr(optimizer, lr, epoch_index)

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            # get label vectors
            y_vec_train_np = labels2Vec(target, label_dict, output_dim)
            y_vec_train = torch.from_numpy(y_vec_train_np)

            target_np = target.numpy()
            sample_weight = []
            for y in target_np:
                sample_weight.append(counter[y])
            sample_weight = torch.from_numpy(np.array(sample_weight))
            sample_weight = sample_weight.type(torch.FloatTensor)
            sample_weight = sample_weight.cuda()
            sample_weight = Variable(sample_weight)

            if useCuda:  # use GPU
                x, y_vec_train = x.cuda(), y_vec_train.cuda()
            test_x = Variable(x, volatile=True)
            x, y_vec_train = Variable(x), Variable(y_vec_train)

            # # get response from old heads
            old_outputs = []
            old_mid_out = old_model.get_middle_output(x)
            for head_idx in range(len(model.head_list) - 1):
                old_output = old_model.get_output(old_mid_out, head_idx)
                old_outputs.append(old_output)

            # calculate old loss
            old_loss = Variable(torch.zeros(1).cuda())
            new_mid_out = model.get_middle_output(x)
            for idx in range(len(model.head_list) - 1):
                out = model.get_output(new_mid_out, idx)
                old_loss += ceriation(out, old_outputs[idx])  # response consolidation loss

            # calculate new loss
            new_loss = Variable(torch.zeros(1).cuda())
            for head_idx in range(len(model.head_list)):
                out = model.get_output(new_mid_out, head_idx)
                new_loss += ceriation(out, y_vec_train, sample_weight=None)

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

        acc = inference(model, test_loader, label_dict, useCuda=True, k=1)

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
                # best_new_acc = new_acc
                # best_old_acc = old_acc

    print("best acc:", best_acc)
    # print("best new acc:", best_new_acc)
    # print("best old acc:", best_old_acc)

def inference(model, test_loader, label_dict, useCuda=True, k=1):

    correct_cnt, sum_loss, total_cnt = 0, 0, 0

    if useCuda:
        model = model.cuda()
    model.eval()

    head_num = len(model.head_list)

    for batch_idx, (x, target) in enumerate(test_loader):

        x = Variable(x, volatile=True)

        if useCuda:
            x = x.cuda()

        sim_list = []

        mid_out = model.get_middle_output(x)

        for head_idx in range(0, head_num):

            start = head_idx * CLASS_NUM_IN_BATCH
            label_vec_index = [i for i in range(start, start+CLASS_NUM_IN_BATCH)]
            label_vecs = [label_dict[idx] for idx in label_vec_index]
            label_vecs = np.array(label_vecs).squeeze(axis=1)  # (CLASS_NUM_IN_BATCH, 100)
            label_vecs = label_vecs.reshape((CLASS_NUM_IN_BATCH, -1))  # (CLASS_NUM_IN_BATCH, 100)

            pred_vecs = model.get_output(mid_out, head_idx)
            pred_vecs = pred_vecs.cpu().data.numpy()
            sim = cosine_similarity(pred_vecs, label_vecs)
            sim_list.append(sim)

        total_cnt += x.data.size()[0]

        sim_array = np.hstack(sim_list)
        target_np = target.numpy()
        correct_cnt += calc_TopK_Acc(sim_array, target_np, topk=k)

        # pred_label = np.argmax(sim_array, axis=1)
        # correct_cnt += (pred_label == target.numpy()).sum()

    acc = (correct_cnt * 1.0 / float(total_cnt))
    print("inference acc:", acc)
    return acc


if __name__ == '__main__':

    # parameters
    TOTAL_CLASS_NUM = 10
    CLASS_NUM_IN_BATCH = 2
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    batch_size = 128
    epoch = 50
    lr = 0.01
    label_set = set()
    label_dict = {}
    output_dim = 100
    lamda = 2
    threshold = 0.15
    sampleNum = 50

    # data root
    train_root = './data/CIFAR10_png/training/'
    test_root = './data/CIFAR10_png/testing/'
    data_pool_root = './data/CIFAR10_png/data_pool/'

    # train_root = './data/tiny-imagenet-200_1/training/'
    # test_root = './data/tiny-imagenet-200_1/testing/'
    # data_pool_root = './data/tiny-imagenet-200_1/data_pool/'

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

    print('==> Building model..')
    # net = Simple_CNN_Multi_Head_LMRC(output_dim=output_dim, normalize=True)
    net = ResNet18_LMRC(outputDim=output_dim, normalize=True)

    for i in range(0, TOTAL_CLASS_NUM, CLASS_NUM_IN_BATCH):  

        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])

 
        if i == 0:  
            trainLoader, train_classes = load_cifar10_online([train_root],
                                                             category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                             batchSize=batch_size, train=True, shuffle=True)
        else: 
            trainLoader, train_classes = load_cifar10_online([train_root],
                                                             category_indexs=class_index[i:i + CLASS_NUM_IN_BATCH],
                                                             batchSize=batch_size, data_pool_root=data_pool_root, train=True, shuffle=True)
        testLoader, test_classes = load_cifar10_online([test_root],
                                                       category_indexs=class_index[0:i + CLASS_NUM_IN_BATCH],
                                                       batchSize=batch_size, train=False, shuffle=False)
        print("train classes:", train_classes)
        print("test classes:", test_classes)

        # assign label vector
        label_set, label_dict, new_label = label_allotter(label_set, train_classes, label_dict, output_dim, threshold)
        print("label set:", label_set)
        print("label dict keys:", label_dict.keys())
        print("new classes:", new_label)

        # add head
        head_index = net.add_head_layer()

        # train and save model
        saveModelPath = "./model/cifar10_online/resnet18_label_mapping_multi_head_" + str(i)

        train(model=net, cur_class=i, head_index=head_index, lamda=lamda, epoch=epoch, lr=lr, output_dim=output_dim, train_loader=trainLoader
              , test_loader=testLoader, label_dict=label_dict, modelPath=saveModelPath, checkPoint=10,
              useCuda=use_cuda, adjustLR=True, earlyStop=False, tolearnce=4)

        net, best_acc, best_epoch = loadModel(saveModelPath, net)

        # random sampling to update data pool
        new_classes = class_index[i: i + CLASS_NUM_IN_BATCH]
        for cls_idx in new_classes:
            random_sample_data_pool(root_dir=train_root, data_pool_root=data_pool_root, class_index=cls_idx,
                                    sample_num=sampleNum)




