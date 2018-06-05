from torch.utils.data import Dataset
import csv
import cv2
import numpy
from math import floor
from homographydataset import HomographyDataset


class SpiralSequence():

    def __init__(self, file, translate=10, crop_size=(256, 256), patch_size=(128, 128), patch_location = (128, 128)):

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.translate = translate

        self.image = cv2.imread(file)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.resize(self.image, (800,800))

        self.i = 0

    def update_index(self):




        if self.dir == 3:
            self.h += 1
            if self.h >= self.r+1:
                self.dir += 1
        elif self.dir == 0 or self.dir == 4:
            self.v += 1
            if self.v >= self.r+1:
                self.dir += 1

        elif self.dir == 1:
            self.h -= 1
            if self.h <= -(self.r+1):
                self.dir += 1
        elif self.dir == 2:
            self.v -= 1
            if self.v <= -(self.r+1):
                self.dir += 1

        if self.dir == 6:
           self.r += 1
           self.dir = 0


        if self.dir == 5:
            self.dir = 0
            self.r +=1


    def __iter__(self):
        self.h = 0
        self.v = 0
        self.dir = 6
        self.r = 0
        return self

    def next(self):

        self.center = (self.image.shape[0]/2, self.image.shape[1]/2)

        image1 = self.image[self.center[0] + self.v * self.translate : self.center[0] + self.v * self.translate + self.crop_size[1], self.center[1] + self.h * self.translate : self.center[1] + self.h * self.translate + self.crop_size[1]]

        patch1 = image1[self.patch_location[0] - self.patch_size[0] / 2:self.patch_location[0] + self.patch_size[0] / 2,
                 self.patch_location[1] - self.patch_size[1] / 2:self.patch_location[1] + self.patch_size[1] / 2]


        self.update_index()

        return image1,patch1