from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # TODO: Salima: change to CNN
        self.conv1 = nn.Conv2d(
            in_channels  = 1,
            out_channels = 4,
            kernel_size  = (3,3),
            stride       = 1,
            padding      = 1 
        )
        self.conv2 = nn.Conv2d(
            in_channels  = 4,
            out_channels = 4,
            kernel_size  = (3,3),
            stride       = 1,
            padding      = 1 
        )
        dim_mu = 640
        dim_logvar = 640
        self.fc1 = nn.Linear(4*32*32, dim_mu)
        self.fc2 = nn.Linear(4*32*32, dim_logvar)

        self.fc3 = nn.Linear(dim_mu, 4*32*32)
        self.conv3 = nn.Conv2d(
            in_channels  = 4,
            out_channels = 4,
            kernel_size  = (3,3),
            stride       = 1,
            padding      = 1 
        )
        self.conv4 = nn.Conv2d(
            in_channels  = 4,
            out_channels = 1,
            kernel_size  = (3,3),
            stride       = 1,
            padding      = 1 
        )

        #self.fc1 = nn.Linear(1024, 400) # input 1024
        #self.fc21 = nn.Linear(400, 20)
        #self.fc22 = nn.Linear(400, 20)
        #self.fc3 = nn.Linear(20, 400)
        #self.fc4 = nn.Linear(400, 1024) # decode ouoput  = input = 1024

    def encode(self, x): # x.shape = (batch, 32, 32)
        h1 = self.conv1(x) # h1.shape = (batch, 4, 32, 32)
        h1 = F.relu(h1) # h1.shape = (batch, 4, 32, 32)
        h2 = self.conv2(h1) # h2.shape = (batch, 4, 32, 32)
        h2 = F.relu(h2) # h2.shape = (batch, 4, 32, 32)
        return self.fc1(h2.view(-1, 4*32*32)), self.fc2(h2.view(-1, 4*32*32))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z)) # h3.shape = (batch, 4*32*32)
        h3 = h3.view(-1, 4, 32, 32) # h3.shape = (batch, channels =  4, 32, 32)
        h4 = F.relu(self.conv3(h3))  # h4.shape = (batch, 4, 32, 32)
        h5 = torch.sigmoid(self.conv4(h4))   # h5.shape = (batch, 1, 32, 32)
        return h5

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        mu, logvar = self.encode(x) # x.shape = (batch, 1, 32, 32)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

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

class Wide_ResNet2(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, data_shape):
        super(Wide_ResNet2, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 16
        # print("num_classes",num_classes)
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
        # self.bn2 = nn.BatchNorm1d(num_classes, momentum=0.9)
        w = data_shape[0]
        h = data_shape[1]
        
        n_output_bn =int( ((w//4 - 8)//8 + 1) * ((h//4 - 8)//8 + 1) * 640)
        print("n_output_bn",n_output_bn)
        self.linear1 = nn.Linear(int(n_output_bn), int(n_output_bn/2))
        self.bn2 = nn.BatchNorm1d(int(n_output_bn/2), momentum=0.9)
        self.linear2 = nn.Linear(int(n_output_bn/2), int(n_output_bn/4))
        self.bn3 = nn.BatchNorm1d(int(n_output_bn/4), momentum=0.9)
        self.linear3 = nn.Linear(int(n_output_bn/4), self.num_classes)
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
        # print('before max_pool: ', out.shape)

        vec = F.max_pool2d(out, 8)
        out = vec.view(vec.size(0), -1)
        # print('after max_pool: ', out.shape)
        out = self.linear1(out)
        # print('linear1: ', out.shape)
        out = F.relu(self.bn2(out))
        # print('linear2: ', out.shape)
        out = self.linear2(out)
        out = F.relu(self.bn3(out))
        out = self.linear3(out)
        
        return out, vec
class Wide_ResNet_preMixup(nn.Module):
   def __init__(self, depth, widen_factor, dropout_rate, num_classes):
       super(Wide_ResNet_preMixup, self).__init__()
       self.in_planes = 16
       # print("num_classes",num_classes)
       assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
       n = (depth-4)/6
       k = widen_factor

    #    print('| Wide-Resnet %dx%d' %(depth, k))
       nStages = [16, 16*k, 32*k, 64*k]

       self.conv1 = conv3x3(1,nStages[0])
       self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)

   def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
       strides = [stride] + [1]*(int(num_blocks)-1)
       layers = []

       for stride in strides:
           layers.append(block(self.in_planes, planes, dropout_rate, stride))
           self.in_planes = planes

       return nn.Sequential(*layers)

   def forward(self, x):
       # print('Initial shape: ', x.shape)\
       # set_trace()
       out = self.conv1(x)
       # print('Conv1: ', out.shape)
       out = self.layer1(out)
       return out
       

class Wide_ResNet_postMixup(nn.Module):
   def __init__(self, depth, widen_factor, dropout_rate, num_classes):
       super(Wide_ResNet_postMixup, self).__init__()
       
       # print("num_classes",num_classes)
       assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
       n = (depth-4)/6
       k = widen_factor
       self.in_planes = 16*k # 64  16

    #    print('| Wide-Resnet %dx%d' %(depth, k))
       nStages = [16, 16*k, 32*k, 64*k]

       self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
       self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
       self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
       self.linear = nn.Linear(640, num_classes)


   def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
       strides = [stride] + [1]*(int(num_blocks)-1)
       layers = []

       for stride in strides:
           
           layers.append(block(self.in_planes, planes, dropout_rate, stride))
           self.in_planes = planes

       return nn.Sequential(*layers)

   def forward(self, x):
       
       #set_trace()
       # print('layer1: ', out.shape)
       out = self.layer2(x)

       # print('layer2: ', out.shape)
       out = self.layer3(out)
       # print('layer3: ', out.shape)
       out = F.relu(self.bn1(out))
       # print('before max_pool: ', out.shape)

       vec = F.max_pool2d(out, 8)
       out = vec.view(vec.size(0), -1)
       # print('after max_pool: ', out.shape)
       out = self.linear(out)
       
       return out, vec



if __name__ == '__main__':
    net=Wide_ResNet2(10, 4, 0.3, 7)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())

if __name__ == '__main__':
    net=Wide_ResNet2(10, 4, 0.3, 7)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())