import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
from torch import autograd
import numpy as np


class Simple_CNN_fine_tune(nn.Module):

    def __init__(self, outputDim):

        super(Simple_CNN_fine_tune, self).__init__()

        self.outputDim = outputDim

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)

        setattr(self, 'final_fc', nn.Linear(128, outputDim).cuda())

        # Self Params
        self.params = [param for param in self.parameters()]

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5)
        out = self.final_fc(out)
        return out

    def change_output_dim(self, new_dim):

        setattr(self, "final_fc", nn.Linear(128, new_dim).cuda())

    def get_output_dim(self):

        return self.final_fc.out_features

