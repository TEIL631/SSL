import torch.nn as nn
import torch.nn.functional as F

class SimpleNN1(nn.Module):
    def __init__(self, n_class, data_shape):
        super(SimpleNN1, self).__init__()
        w = data_shape[0]
        h = data_shape[1]
        n_output_bn = ((w//4 - 8)//8 + 1) * ((h//4 - 8)//8 + 1) * 640
        self.output = nn.Linear(n_output_bn+640, n_class)
    def forward(self, x):
        x = self.output(x)
        return x

class SimpleNN3(nn.Module):
    def __init__(self, n_class, data_shape):
        super(SimpleNN3, self).__init__()
        w = data_shape[0]
        h = data_shape[1]
        n_output_bn = ((w//4 - 8)//8 + 1) * ((h//4 - 8)//8 + 1) * 640
        self.output = nn.Linear(int(n_output_bn+640), int((n_output_bn+640)/2))
        self.bn1 = nn.BatchNorm1d(int((n_output_bn+640)/2), momentum=0.9)
        self.output2 = nn.Linear(int((n_output_bn+640)/2), int((n_output_bn+640)/4))
        self.bn2 = nn.BatchNorm1d(int((n_output_bn+640)/4), momentum=0.9)
        self.output3 = nn.Linear(int((n_output_bn+640)/4), n_class)
    def forward(self, x):
        x = self.output(x)
        x = F.relu(self.bn1(x))
        x = self.output2(x)
        x = F.relu(self.bn2(x))
        x = self.output3(x)
        return x