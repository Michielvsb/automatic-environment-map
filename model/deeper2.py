import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class HomographyNet(nn.Module):

    def __init__(self):
        super(HomographyNet, self).__init__()

        self.batchnorm1 = nn.BatchNorm2d(2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.batchnorm6 = nn.BatchNorm2d(128)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.batchnorm8 = nn.BatchNorm2d(256)
        self.batchnorm9 = nn.BatchNorm1d(65536)
        self.batchnorm10 = nn.BatchNorm1d(1024)

        self.conv1_64 = nn.Conv2d(2,32,3,padding=(1,1))

        self.conv2_64 = nn.Conv2d(32, 64, 3,padding=(1,1))

        self.max_pooling1 = nn.MaxPool2d((2,2), stride=2)

        self.conv3_64 = nn.Conv2d(64,64,3,padding=(1,1))

        self.conv4_64 = nn.Conv2d(64,128,3,padding=(1,1))
        self.max_pooling2 = nn.MaxPool2d((2,2), stride=2)

        self.conv5_128 = nn.Conv2d(128,128,3,padding=(1,1))

        self.conv6_128 = nn.Conv2d(128,128,3,padding=(1,1))
        self.max_pooling3 = nn.MaxPool2d((2,2), stride=2)

        self.conv7_128 = nn.Conv2d(128,256,3,padding=(1,1))

        self.conv8_128 = nn.Conv2d(256,256,3,padding=(1,1))

        self.lin1 = nn.Linear(16*16*256,1024)

        self.lin2 = nn.Linear(1024,8)


    def forward(self, x):
        x = F.relu(self.conv1_64(self.batchnorm1(x)))
        x = F.relu(self.conv2_64(self.batchnorm2(x)))
        x = self.max_pooling1(x)
        x = F.relu(self.conv3_64(self.batchnorm3(x)))
        x = F.relu(self.conv4_64(self.batchnorm4(x)))
        x = self.max_pooling2(x)
        x = F.relu(self.conv5_128(self.batchnorm5(x)))
        x = F.relu(self.conv6_128(self.batchnorm6(x)))
        x = self.max_pooling3(x)
        x = F.relu(self.conv7_128(self.batchnorm7(x)))
        x = F.relu(self.conv8_128(self.batchnorm8(x)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.lin1(self.batchnorm9(x)))

        x = self.lin2(self.batchnorm10(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x