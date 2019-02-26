import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import numpy as np


def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = F.log_softmax(logits/T,)   # compute the log of softmax values
    labels = F.softmax(labels/T,)
    loss = -torch.mean(outputs * labels)
    return Variable(loss.data, requires_grad=True).cuda()


class Simple_CNN_LWF(nn.Module):

    def __init__(self, outputDim):

        super(Simple_CNN_LWF, self).__init__()

        self.outputDim = outputDim

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)

        setattr(self, 'final_fc', nn.Linear(128, outputDim).cuda())

        # Self Params
        self.params = [param for param in self.parameters()]

    def forward(self, x, feature=False):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if feature:
            return out
        out = F.dropout(out, p=0.5)
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
        self.n_classes = new_dim

    def freeze_weight(self):

        for param in self.parameters():
            param.requires_grad = False

