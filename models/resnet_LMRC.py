'''
LMRC model in PyTorch.
'''

import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet, BasicBlock, Bottleneck


class ResNet_LMRC(ResNet):
    def __init__(self, block, num_blocks, output_dim=10, normalize=True):
        super(ResNet_LMRC, self).__init__(block, num_blocks, output_dim)
        self.block = block
        self.head_list = []
        self.normalize = normalize
        self.output_dim = output_dim

    def get_middle_output(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out

    def get_output(self, mid_out, head_index):

        out = self.head_list[head_index](mid_out)
        if self.normalize:
            out = F.normalize(out)
        return out

    def forward(self, x, head_index):

        mid_out = self.get_middle_output(x)
        out = self.get_output(mid_out, head_index)

        return out

    def add_head_layer(self):

        curr_head_num = len(self.head_list)
        setattr(self, "head_" + str(curr_head_num + 1), nn.Linear(512*self.block.expansion, self.output_dim).cuda())
        self.head_list.append(getattr(self, "head_"+str(curr_head_num + 1)))
        print("current head num:", len(self.head_list))
        return len(self.head_list) - 1

    def get_output_dim(self):

        return len(self.head_list)

    def freeze_weight(self):

        for param in self.parameters():
            param.requires_grad = False


def ResNet18_LMRC(outputDim, normalize=True):
    return ResNet_LMRC(BasicBlock, [2,2,2,2], outputDim, normalize)


def ResNet34_LMRC(outputDim, normalize=True):
    return ResNet_LMRC(BasicBlock, [3,4,6,3], outputDim, normalize)


def ResNet50_LMRC(outputDim, normalize=True):
    return ResNet_LMRC(Bottleneck, [3,4,6,3], outputDim, normalize)


def ResNet101(outputDim, normalize=True):
    return ResNet_LMRC(Bottleneck, [3,4,23,3], outputDim, normalize)


def ResNet152(outputDim, normalize=True):
    return ResNet_LMRC(Bottleneck, [3,8,36,3], outputDim, normalize)
