import torch

class UnsupervisedHomographyNet(torch.nn.Module):

    def __init__(self, patch_size, patch_location, net_name, map_name="homographymap"):
        super(UnsupervisedHomographyNet, self).__init__()

        net = __import__("model."+net_name, fromlist="HomographyNet")
        print "Imported model."+net_name
        map = __import__("model." + map_name, fromlist="HomographyMap")
        print "Imported model." + map_name

        self.homographynet = net.HomographyNet()

        self.homographymap = map.HomographyMap(patch_size, patch_location)

    def forward(self, *input):
        x = input[0]
        image = input[1]
        x = self.homographynet(x)
        h, transformed_image = self.homographymap(x,image)

        return transformed_image, h, x

    def forward_net(self, *input):
        x = input[0]
        x = self.homographynet(x)

        return x