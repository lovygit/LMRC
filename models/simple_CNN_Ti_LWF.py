import torch.nn as nn
import torch
import torch.nn.functional as F


class Simple_CNN_Multi_Head_LWF(nn.Module):

    def __init__(self, output_dim=10):
        super(Simple_CNN_Multi_Head_LWF, self).__init__()

        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        # self.head_1 = nn.Linear(128, output_dim)
        self.head_list = []

    def forward(self, x, head_index):

        assert len(self.head_list) != 0

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.dropout(out, p=0.5)
        out = self.head_list[head_index](out)
        return out

    def add_head_layer(self, add_dim):

        curr_head_num = len(self.head_list)
        setattr(self, "head_" + str(curr_head_num + 1), nn.Linear(128, add_dim).cuda())
        self.head_list.append(getattr(self, "head_"+str(curr_head_num + 1)))
        print("current head num:", len(self.head_list))
        return len(self.head_list) - 1

    def get_total_output(self, x):

        assert len(self.head_list) != 0
        total_out = []
        for i in range(len(self.head_list)):
            out = self.forward(x, i)
            total_out.append(out)
        total_out = torch.cat(total_out, dim=1).cuda()

        return total_out

    def freeze_weight(self):

        for param in self.parameters():
            param.requires_grad = False
