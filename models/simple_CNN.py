
import torch.nn as nn
import torch.nn.functional as F


class Simple_CNN(nn.Module):
    def __init__(self, ouputDim=10, normalize=False):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1   = nn.Linear(64*5*5, 128)
        self.fc2   = nn.Linear(128, ouputDim)
        self.normalize = normalize

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.dropout(out, p=0.5)
        out = self.fc2(out)
        if self.normalize:
            out = F.normalize(out)
        return out
