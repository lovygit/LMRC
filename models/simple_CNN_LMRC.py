
import torch.nn as nn
import torch.nn.functional as F


class Simple_CNN_LMRC(nn.Module):

    def __init__(self, output_dim=10, normalize=True):
        super(Simple_CNN_LMRC, self).__init__()

        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        # self.head_1 = nn.Linear(128, output_dim)
        self.head_list = []
        self.normalize = normalize

    def add_head_layer(self):

        curr_head_num = len(self.head_list)
        setattr(self, "head_" + str(curr_head_num + 1), nn.Linear(128, self.output_dim).cuda())
        self.head_list.append(getattr(self, "head_"+str(curr_head_num + 1)))
        print("current head num:", len(self.head_list))
        return len(self.head_list) - 1

    def get_middle_output(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.dropout(out, p=0.5)

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

    def freeze_weight(self):

        for param in self.parameters():
            param.requires_grad = False
