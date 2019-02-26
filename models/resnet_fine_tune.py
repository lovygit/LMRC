'''
resnet fine tuning model, only reinit the top linear layer and fine tune

'''
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet, BasicBlock, Bottleneck


class ResNet_Fine_Tuning(ResNet):
    def __init__(self, block, num_blocks, output_dim=10):
        super(ResNet_Fine_Tuning, self).__init__(block, num_blocks, output_dim)
        self.block = block
        self.final_fc = nn.Linear(512 * block.expansion, output_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out) # cifar10 no maxpool
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.final_fc(out)
        return out

    def change_output_dim(self, new_dim):

        setattr(self, "final_fc", nn.Linear(512*self.block.expansion, new_dim).cuda())

    def get_output_dim(self):

        return self.final_fc.out_features


def ResNet18_fine_tune(outputDim):
    return ResNet_Fine_Tuning(BasicBlock, [2,2,2,2], outputDim)


def ResNet34_fine_tune(outputDim):
    return ResNet_Fine_Tuning(BasicBlock, [3,4,6,3], outputDim)


def ResNet50_fine_tune(outputDim):
    return ResNet_Fine_Tuning(Bottleneck, [3,4,6,3], outputDim)


def ResNet101(outputDim):
    return ResNet_Fine_Tuning(Bottleneck, [3,4,23,3], outputDim)


def ResNet152(outputDim):
    return ResNet_Fine_Tuning(Bottleneck, [3,8,36,3], outputDim)
