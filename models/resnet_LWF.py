'''
LWF model in PyTorch.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.resnet import ResNet, BasicBlock, Bottleneck


def MultiClassCrossEntropy(logits, labels, T):

    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = F.log_softmax(logits/T,)   # compute the log of softmax values
    labels = F.softmax(labels/T,)
    loss = -torch.mean(outputs * labels)  # should be mean because the cross entropy is mean
    return Variable(loss.data, requires_grad=True).cuda()


class ResNet_LWF(ResNet):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_LWF, self).__init__(block, num_blocks, num_classes)

        self.block = block
        setattr(self, 'final_fc', nn.Linear(512*block.expansion, num_classes).cuda())

    def forward(self, x, feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if feature:
            return out
        out = self.final_fc(out)
        return out

    def get_output_dim(self):

        return self.final_fc.out_features

    def change_output_dim(self, new_dim):

        in_features = self.final_fc.in_features
        out_features = self.final_fc.out_features
        weight = self.final_fc.weight.data

        new_out_features = new_dim
        self.final_fc = nn.Linear(in_features, new_out_features, bias=False)

        self.final_fc.weight.data[:out_features] = weight
        self.n_classes = new_out_features

    def freeze_weight(self):

        for param in self.parameters():
            param.requires_grad = False


def ResNet18_LWF(outputDim):
    return ResNet_LWF(BasicBlock, [2,2,2,2], outputDim)


def ResNet34_LWF(outputDim):
    return ResNet_LWF(BasicBlock, [3,4,6,3], outputDim)


def ResNet50_LWF(outputDim):
    return ResNet_LWF(Bottleneck, [3,4,6,3], outputDim)


def ResNet101_LWF(outputDim):
    return ResNet_LWF(Bottleneck, [3,4,23,3], outputDim)


def ResNet152_LWF(outputDim):
    return ResNet_LWF(Bottleneck, [3,8,36,3], outputDim)
