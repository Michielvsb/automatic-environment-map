from torch.utils.data import Dataset
import csv
import cv2
import numpy
from math import floor
from homographydataset import HomographyDataset


class PanoramaSequence():

    def __init__(self, file, translate=10, crop_size=(256, 256), patch_size=(128, 128), patch_location = (128, 128)):

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.translate = translate

        self.image = cv2.imread(file)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def next(self):

        if self.i > int(floor(float(self.image.shape[1]-self.crop_size[1]) / self.translate)):
            raise StopIteration

        else:

            image1 = self.image[0:256, self.i * self.translate:self.i * self.translate + self.crop_size[1]]

            patch1 = image1[self.patch_location[0] - self.patch_size[0] / 2:self.patch_location[0] + self.patch_size[0] / 2,
                     self.patch_location[1] - self.patch_size[1] / 2:self.patch_location[1] + self.patch_size[1] / 2]


            self.i += 1

        return image1,patch1