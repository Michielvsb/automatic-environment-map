import torch.nn as nn
from resnet import resnet18

class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.model = resnet18(pretrained=False, num_classes=8)

    def forward(self, x):
        return self.model(x)
