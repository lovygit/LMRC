'''
EWC model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from models.resnet import ResNet, BasicBlock, Bottleneck
from torch.autograd import Variable as V


class ResNet_EWC(ResNet):
    def __init__(self, block, num_blocks, outputDim=10):
        super(ResNet_EWC, self).__init__(block, num_blocks, outputDim)
        self.block = block
        self.final_fc_list = []
        self.in_planes = 64
        self.outputDim = outputDim

        for i in range(self.outputDim):
            setattr(self, 'final_'+str(i), nn.Linear(512*block.expansion, 1).cuda())
            self.final_fc_list.append(getattr(self, 'final_'+str(i)))

        # Init Matrix which will get Fisher Matrix
        self.Fisher = {}

        self.params = [param for param in self.parameters()]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)  # cifar10 no maxpool
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        final = []

        for idx in range(len(self.final_fc_list)):
            fc_out = getattr(self, "final_" + str(idx))(out)
            final.append(fc_out)
        out = torch.cat(final, dim=1)  # expand the dimension
        return out

    def add_output_dim(self, add_dim=1):

        cur_final_index = len(self.final_fc_list)-1

        setattr(self, "final_"+str(cur_final_index+1), nn.Linear(512*self.block.expansion, add_dim).cuda())

        self.final_fc_list.append(getattr(self, "final_"+str(cur_final_index+1)))

    def get_output_dim(self):

        curr_output_dim = 0
        for fc in self.final_fc_list:
            curr_output_dim += fc.out_features
        return curr_output_dim

    def estimate_fisher(self, dataset, sample_size, batch_size):

        data_loader = dataset
        loglikelihoods = []
        loglikelihood_grads = None

        for x, y in data_loader:
            x = V(x).cuda() if self._is_on_cuda() else V(x)
            y = V(y).cuda() if self._is_on_cuda() else V(y)

            pred = self(x)

            log_prob = F.log_softmax(pred)

            loglikelihoods.append(log_prob[range(batch_size), y.data])

            if len(loglikelihoods) >= sample_size // batch_size:
                break

            loglikelihood = torch.cat(loglikelihoods).mean(0)

            loglikelihood_grads = autograd.grad(loglikelihood, self.parameters(), retain_graph=True)

        parameter_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]

        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            self.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data)

    def ewc_loss(self, lamda, cuda=False):

        losses = [V(torch.zeros(1)).cuda() if cuda else V(torch.zeros(1))]

        for n, p in self.named_parameters():

            n = n.replace('.', '__')

            try:  
                mean = getattr(self, '{}_estimated_mean'.format(n))
                fisher = getattr(self, '{}_estimated_fisher'.format(n))
                mean = V(mean)
                fisher = V(fisher)
                losses.append((fisher * (p - mean) ** 2).sum())
            except:
                continue

        return (lamda/2)*sum(losses)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


def ResNet18_EWC(outputDim):
    return ResNet_EWC(BasicBlock, [2,2,2,2], outputDim)


def ResNet34_EWC(outputDim):
    return ResNet_EWC(BasicBlock, [3,4,6,3], outputDim)


def ResNet50_EWC(outputDim):
    return ResNet_EWC(Bottleneck, [3,4,6,3], outputDim)


def ResNet101_EWC(outputDim):
    return ResNet_EWC(Bottleneck, [3,4,23,3], outputDim)


def ResNet152_EWC(outputDim):
    return ResNet_EWC(Bottleneck, [3,8,36,3], outputDim)
