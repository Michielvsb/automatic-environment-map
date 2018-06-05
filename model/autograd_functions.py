from torch.autograd import Function, Variable
import torch
from math import pow, floor, ceil

class MapImage(Function):

    @staticmethod
    def forward(ctx, grid, image):

        ctx.grid = grid

        grid = grid.repeat(1,1,1,2)
        batch_size, height, width = grid.size(0), grid.size(1), grid.size(2)

        index = torch.zeros(batch_size, height, width, 4).cuda()
        weights = torch.zeros(batch_size, height, width, 4).cuda()

        index[:, :, :, 0:2] = torch.floor(grid[:, :, :, 0:2])  # co_w_fl, co_h_fl
        index[:, :, :, 2:4] = torch.ceil(grid[:, :, :, 0:2])  # co_w_ce, co_h_ce

        weights[:, :, :, [0, 2]] = (1 - torch.abs(grid[:, :, :, 0:1] - index[:, :, :, 0:1])).repeat(1, 1, 1, 2)  # 1-abs(co_w-co_w_fl)
        weights[:, :, :, [1, 3]] = torch.remainder((1 - torch.abs(grid[:, :, :, 0:1] - index[:, :, :, 2:3])).repeat(1, 1, 1, 2), 1)  # 1-abs(co_w-co_w_ce)

        weights[:, :, :, 0:2] = weights[:, :, :, 0:2] * (1 - torch.abs(grid[:, :, :, 1:2] - index[:, :, :, 1:2])).repeat(1, 1, 1, 2) # 1-abs(co_h-co_h_fl)
        weights[:, :, :, 2:4] = weights[:, :, :, 2:4] * torch.remainder((1 - torch.abs(grid[:, :, :, 1:2] - index[:, :, :, 3:4])).repeat(1, 1, 1, 2), 1)  # 1-abs(co_h-co_h_ce)

        # index [:, :, :, 0] h_fl, w_fl | [:, :, :, 0] h_fl, w_ce | [:, :, :, 2] h_ce, w_fl | [:, :, :, 3] h_ce, w_ce |

        b = torch.arange(batch_size).cuda().unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, height, width, 2)*height*width*4

        temp_index = index.clone()
        index[:, :, :, [0, 2]] = b + image.shape[2] * temp_index[:, :, :, [1, 3]] + temp_index[:, :, :, 0:1].repeat(1, 1, 1, 2) # image[b, cor_h_fl, cor_w_fl], image[b, cor_h_ce, cor_w_fl]
        index[:, :, :, [1, 3]] = b + image.shape[2] * temp_index[:, :, :, [1, 3]] + temp_index[:, :, :, 2:3].repeat(1, 1, 1, 2) # image[b, cor_h_fl, cor_w_ce], image[b, cor_h_ce, cor_w_ce]

        index = index.view(batch_size, -1, 4)  # make 1D in height/width
        index = torch.clamp(index, 0, batch_size*image.shape[1]*image.shape[2]-1)

        image_layers = torch.take(image.float(), index.long())
        image_layers = image_layers.view(batch_size, height, width, 4)

        transformed_image = image_layers * weights
        transformed_image = torch.sum(transformed_image, 3)

        ctx.image_layers = image_layers.clone()

        ctx.image = image

        return transformed_image

    # @staticmethod
    # def backward(ctx, grad_output):
    #
    #     grad_output = grad_output.data
    #
    #     grid, image = ctx.grid, ctx.image
    #
    #     batch_size = grid.size(0)
    #
    #     height, width = grid.size(1), grid.size(2)
    #
    #     grad_input = torch.zeros((batch_size, height, width, 2))
    #     for b in range(batch_size):
    #         for u in range(width):
    #             for v in range(height):
    #
    #                 cor_h, cor_w = grid[b, v, u, 1], grid[b, v, u, 0]
    #                 cor_h_fl, cor_h_ce, cor_w_fl, cor_w_ce = int(floor(cor_h)), int(ceil(cor_h)), int(floor(cor_w)), int(ceil(cor_w))
    #
    #                 grad_input[b, v, u, 0] -= image[b, cor_h_fl, cor_w_fl]*(1-abs(cor_w-cor_w_fl))  # d_image/dv
    #                 grad_input[b, v, u, 0] -= image[b, cor_h_fl, cor_w_ce]*(1-abs(cor_w-cor_w_ce))  # d_image/dv
    #                 grad_input[b, v, u, 0] += image[b, cor_h_ce, cor_w_fl]*(1-abs(cor_w-cor_w_fl))  # d_image/dv
    #                 grad_input[b, v, u, 0] += image[b, cor_h_ce, cor_w_ce]*(1-abs(cor_w-cor_w_ce))  # d_image/dv
    #                 grad_input[b, v, u, 0] *= grad_output[b, v, u]
    #
    #                 grad_input[b, v, u, 1] -= image[b, cor_h_fl, cor_w_fl]*(1-abs(cor_h-cor_h_fl))  # d_image/du
    #                 grad_input[b, v, u, 1] += image[b, cor_h_fl, cor_w_ce]*(1-abs(cor_h-cor_h_fl))  # d_image/du
    #                 grad_input[b, v, u, 1] -= image[b, cor_h_ce, cor_w_fl]*(1-abs(cor_h-cor_h_ce))  # d_image/du
    #                 grad_input[b, v, u, 1] += image[b, cor_h_ce, cor_w_ce]*(1-abs(cor_h-cor_h_ce))  # d_image/du
    #                 grad_input[b, v, u, 1] *= grad_output[b, v, u]
    #
    #
    #     print grad_input
    #     return Variable(grad_input), None

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.data
        grid, image = ctx.grid, ctx.image
        batch_size = grid.size(0)
        height, width = grid.size(1), grid.size(2)
        image_layers = ctx.image_layers

        image_layers = image_layers.unsqueeze(4)
        image_layers = image_layers.repeat(1,1,1,1,2)


        weights = torch.zeros(batch_size, height, width, 4, 2).cuda()

        index = torch.zeros(batch_size, height, width, 4).cuda()

        index[:, :, :, 0:2] = torch.floor(grid[:, :, :, 0:2])  # co_w_fl, co_h_fl
        index[:, :, :, 2:4] = torch.ceil(grid[:, :, :, 0:2])  # co_w_ce, co_h_ce

        weights[:, :, :, 0, 1] = (torch.abs(grid[:, :, :, 0:1] - index[:, :, :, 0:1]) - 1)  # 1-abs(co_w-co_w_fl)*-1
        weights[:, :, :, 1, 1] = -1*torch.remainder((1 - torch.abs(grid[:, :, :, 0:1] - index[:, :, :, 2:3])) , 1) # 1-abs(co_w-co_w_ce)*-1
        weights[:, :, :, 2:4, 1] = -weights[:, :, :, 0:2, 1]

        weights[:, :, :, 0:2, 0] = (1 - torch.abs(grid[:, :, :, 1:2] - index[:, :, :, 1:2])).repeat(1, 1, 1, 2)  # 1-abs(co_h-co_h_fl)
        weights[:, :, :, 2:4, 0] = torch.remainder((1 - torch.abs(grid[:, :, :, 1:2] - index[:, :, :, 3:4])).repeat(1, 1, 1, 2), 1)  # 1-abs(co_h-co_h_ce)
        weights[:, :, :, 0, 0] = -weights[:, :, :, 0, 0]
        weights[:, :, :, 2, 0] = -weights[:, :, :, 2, 0]

        image_layers = image_layers*weights

        image_layers = torch.sum(image_layers, 3)

        grad_input = image_layers*grad_output.unsqueeze(3).repeat(1,1,1,2)

        return Variable(grad_input), None


class Test(Function):

    @staticmethod
    def forward(ctx, x):
        print "Test forward"
        x = torch.zeros((x.size(0),x.size(1)))
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        print "Test backward"
        print grad_outputs
        return grad_outputs



class Test2(Function):

    @staticmethod
    def forward(ctx, x):
        print "Test2 forward"
        x = torch.ones((x.size(0), x.size(1)))
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        print "Test2 backward"
        print grad_outputs
        return grad_outputs



