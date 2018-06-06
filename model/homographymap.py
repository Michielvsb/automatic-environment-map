import torch
from torch.autograd import Variable
from autograd_functions import Test, Test2,MapImage
import time

class HomographyMap(torch.nn.Module):

    def __init__(self, patch_size, patch_location):
        super(HomographyMap, self).__init__()
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.map_image_f = MapImage()

    def forward(self, h4pt, image):
        H = torch.stack([calculate_h(current_h4pt, self.patch_size, self.patch_location) for current_h4pt in h4pt.split(1)])

        grid = create_grid(H, self.patch_size, self.patch_location)

        transformed_image = self.map_image_f.apply(grid, image)

        return H, transformed_image

def calculate_h(h4pt, patch_size, patch_location):

    corr_o = Variable(torch.cuda.FloatTensor(
        [[patch_location[1] - patch_size[1] / 2, patch_location[0] - patch_size[0] / 2],
         [patch_location[1] - patch_size[1] / 2, patch_location[0] + patch_size[0] / 2],
         [patch_location[1] + patch_size[1] / 2, patch_location[0] + patch_size[0] / 2],
         [patch_location[1] + patch_size[1] / 2, patch_location[0] - patch_size[0] / 2]]))

    corr_d = corr_o + h4pt.view((4, 2))
    H = tdlt(corr_o, corr_d)

    return H

def create_grid(h, size, center):
    height, width = size
    h_center, w_center = center
    batch_size = h.size(0)

    co_h = Variable(torch.arange(h_center - height / 2, h_center + height/2).cuda())
    co_h = co_h.unsqueeze(1) # make a column vector
    co_h = co_h.expand(batch_size, -1, width) # repeat along batch dimension and horizontally
    co_h = co_h.unsqueeze(3)
    co_w = Variable(torch.arange(w_center - width / 2, w_center + width / 2).cuda())
    co_w = co_w.unsqueeze(0) # make a row vector
    co_w = co_w.expand(batch_size, height, -1) # repeat along batch dimension and vertically
    co_w = co_w.unsqueeze(3)
    co = torch.cat([co_w, co_h], 3) # stack in new dimension
    co = co.view((batch_size,-1,2)) # reshape for multiplication
    co = co.transpose(1,2)
    co = torch.cat([co, Variable(torch.ones((batch_size,1,co.size(2))).cuda())],1)

    co = torch.stack([torch.mm(currentH.squeeze(),curr_co.squeeze()) for currentH, curr_co in zip(h.split(1), co.split(1) )])

    co = co / co[:,2:3,:].expand(batch_size, 3, -1) # homogenic coordinates

    grid = co[:, 0:2, :].transpose(1,2).contiguous().view(-1, height, width, 2)

    return grid

def tdlt(corr_o, corr_d): # Tensor direct linear transform
    a = Variable(torch.zeros((4, 3)).cuda())
    b = -corr_o
    c = Variable(-torch.FloatTensor([[1], [1], [1], [1]]).cuda())
    d = torch.unsqueeze(corr_d[:, 1] * corr_o[:, 0],1)
    e = torch.unsqueeze(corr_d[:, 1] * corr_o[:, 1],1)
    f = corr_o
    g = Variable(torch.FloatTensor([[1], [1], [1], [1]]).cuda())
    h = Variable(torch.zeros((4, 3)).cuda())
    i = torch.unsqueeze(-corr_d[:, 0] * corr_o[:, 0],1)
    j = torch.unsqueeze(-corr_d[:, 0] * corr_o[:, 1],1)
    eq1 = torch.cat((a, b, c, d, e), 1)
    eq2 = torch.cat((f, g, h, i, j), 1)

    eq = torch.cat((eq1, eq2), 0)

    bias = torch.cat((-corr_d[:, 1:2], corr_d[:, 0:1]), 0)

    H = eq.inverse().mm(bias)
    H = torch.cat((H, Variable(torch.cuda.FloatTensor([[1]]))), 0)

    return H.view((3, 3))
