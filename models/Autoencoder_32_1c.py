import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def convtrans3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class wide_trans_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        output_padding = 1 if stride==2 else 0
        super(wide_trans_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, output_padding=output_padding, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        return out
        

class Wide_ResNet_autoencoder(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate):
        super(Wide_ResNet_autoencoder, self).__init__()
        self.in_planes = 16
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(1,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        # self.linear = nn.Linear(2560, num_classes)
        self.layer4 = self._wide_layer(wide_trans_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer5 = self._wide_layer(wide_trans_basic, nStages[1], n, dropout_rate, stride=2)
        self.layer6 = self._wide_layer(wide_trans_basic, nStages[0], n, dropout_rate, stride=1)
        self.bn2 = nn.BatchNorm2d(nStages[0], momentum=0.9)
        self.conv2 = convtrans3x3(nStages[0], 1)
        self.bn3 = nn.BatchNorm2d(1, momentum=0.9)
        
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        # print('Initial shape: ', x.shape)
        out = self.conv1(x)
        # print('Conv1: ', out.shape)
        out = self.layer1(out)
        # print('layer1: ', out.shape)
        out = self.layer2(out)
        # print('layer2: ', out.shape)
        out = self.layer3(out)
        # print('layer3: ', out.shape)
        out = F.relu(self.bn1(out))
        # print('before avg_pool: ', out.shape)
        out = F.avg_pool2d(out, 8)
        # print('after avg_pool: ', out.shape)
        hidden_vec = out.view(out.size(0), -1)
        
        out = F.interpolate(out, size=(8, 8))
        # print('after avg_unpool: ', out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer5(out)
        # print(out.shape)
        out = self.layer6(out)
        # print(out.shape)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        # print(out.shape)
        out = torch.tanh(F.relu(self.bn3(out)))
        # print(out.shape)
        
        # print(out.shape)
        return out, hidden_vec
    
# data = np.load('train_3150_data.npy', allow_pickle=True)
# data = data / 2.0
# print(data.shape)
transformer = transforms.Compose(
              [transforms.ToTensor(),
               transforms.Resize((32,32)),
               transforms.RandomHorizontalFlip(),
               transforms.RandomVerticalFlip(),
               transforms.RandomRotation(30)])

# data_tensor = transforms.ToTensor(data)
# new_data_aug = transformer(data)
transformer_2 = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((32,32))])
## Define dataset
class waferDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image1 = transformer(self.data[idx])
        # image1 = torch.squeeze(image1)
        image2 = transformer_2(self.data[idx])
        # image2 = torch.squeeze(image2)
        # print(image1)
        return image1.float().cuda(), image2.float().cuda()
        # return image1.float(), image2.float()


