import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import autograd
import numpy as np


class Simple_CNN_EWC(nn.Module):

    def __init__(self, outputDim):

        super(Simple_CNN_EWC, self).__init__()

        self.outputDim = outputDim

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.final_fc_list = []
        for i in range(self.outputDim):
            setattr(self, 'final_'+str(i), nn.Linear(128, 1).cuda())
            self.final_fc_list.append(getattr(self, 'final_'+str(i)))

        # Init Matrix which will get Fisher Matrix
        self.Fisher = {}

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
        final = []
        for fc_layer in self.final_fc_list:


            fc_out = fc_layer(out)
            final.append(fc_out)
        out = torch.cat(final, dim=1)
        return out

    def add_output_dim(self, add_dim=1):

        cur_final_index = len(self.final_fc_list)-1

        setattr(self, "final_"+str(cur_final_index+1), nn.Linear(128, add_dim).cuda())

        self.final_fc_list.append(getattr(self, "final_"+str(cur_final_index+1)))

    def get_output_dim(self):

        curr_output_dim = 0
        for fc in self.final_fc_list:
            curr_output_dim += fc.out_features
        return curr_output_dim

    def estimate_fisher(self, dataset, sample_size, batch_size):

        # Get loglikelihoods from data
        self.F_accum = []
        for v, _ in enumerate(self.params):
            self.F_accum.append(np.zeros(list(self.params[v].size())))
        data_loader = dataset
        loglikelihoods = []
        loglikelihood_grads = None

        for x, y in data_loader:
            # x = x.view(batch_size, -1)
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
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_estimated_mean'.format(n))
                fisher = getattr(self, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in Vs.
                mean = V(mean)
                fisher = V(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                V(torch.zeros(1)).cuda() if cuda else
                V(torch.zeros(1))
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda