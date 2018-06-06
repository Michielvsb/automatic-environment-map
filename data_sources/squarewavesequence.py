from torch.utils.data import Dataset
import csv
import cv2
import numpy
from math import floor
from random import uniform
from math import radians, cos, sin

class SquareWaveSequence():


    def __init__(self, file, translate=10, amount=1, crop_size=(256, 256), patch_size=(128, 128), patch_location = (128, 128)):

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.translate = translate
        self.amount = amount
        self.image = cv2.imread(file)
        #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.i = 0

        self.h = 0
        self.v = 0
        self.dir = 0
        self.mov = 0
        self.pos = numpy.array([[0,0],[0,0],[0,0],[0,0]], dtype="float32")
        self.rand_h = numpy.array([[1, 0,0],[0,1,0]], dtype="float32")
        self.rotation = 0


    def update_index(self):

        if self.dir == 0 or self.dir == 2:
            self.h += 1
            self.mov += 1

            if self.mov > 9:
                self.dir += 1
                self.mov = 0

        elif self.dir == 3:
            self.v += 1
            self.mov += 1

            if self.mov > 9:
                self.dir = 0
                self.mov = 0


        elif self.dir == 1:
            self.v -= 1
            self.mov += 1

            if self.mov > 9:
                self.dir += 1
                self.mov = 0

    def get(self, i):


        if (i > self.i):
            self.pos = numpy.array(
                [[self.h * self.translate, self.v * self.translate], [self.h * self.translate, self.v * self.translate],
                 [self.h * self.translate, self.v * self.translate], [self.h * self.translate, self.v * self.translate]], dtype="float32")

            self.update_index()
            src = numpy.array([[self.center[1] + self.h * self.translate, self.center[0] + self.v * self.translate],
                               [self.center[1] + self.h * self.translate, self.center[0] + self.v * self.translate + self.crop_size[1]],
                               [self.center[1] + self.h * self.translate + self.crop_size[1], self.center[0] + self.v * self.translate + self.crop_size[1]],
                               [self.center[1] + self.h * self.translate + self.crop_size[1], self.center[0] + self.v * self.translate]], dtype="float32")

            rotation_center = (self.center[1] + self.h * self.translate + (self.crop_size[1] / 2), self.center[0] + self.v * self.translate + (self.crop_size[1] / 2))

            self.rotation = self.rotation+uniform(-self.amount, self.amount)


            self.rand_h = cv2.getRotationMatrix2D(rotation_center, self.rotation, 1)

            self.i = i

        self.center = (self.image.shape[0] / 2, self.image.shape[1] / 2)

        if self.center[0] + self.v * self.translate + self.crop_size[1] > self.image.shape[0] or \
            self.center[0] + self.v * self.translate < 0 or \
            self.center[1] + self.h * self.translate + self.crop_size[1] > self.image.shape[1] or \
            self.center[1] + self.h * self.translate < 0:

            raise StopIteration

        image = cv2.warpAffine(self.image, self.rand_h, (self.image.shape[1], self.image.shape[0]))

        image1 = image[self.center[0] + self.v * self.translate : self.center[0] + self.v * self.translate + self.crop_size[1], self.center[1] + self.h * self.translate : self.center[1] + self.h * self.translate + self.crop_size[1]]

        patch1 = image1[self.patch_location[0] - self.patch_size[0] / 2:self.patch_location[0] + self.patch_size[0] / 2,
                 self.patch_location[1] - self.patch_size[1] / 2:self.patch_location[1] + self.patch_size[1] / 2]

        patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)

        h4pt = numpy.array([[self.h * self.translate, self.v * self.translate], [self.h * self.translate, self.v * self.translate],
                 [self.h * self.translate, self.v * self.translate], [self.h * self.translate, self.v * self.translate]], dtype="float32") - self.pos

        src = numpy.array([[0.0, 0.0], [0.0, self.crop_size[0]],
                 [self.crop_size[1],self.crop_size[0]], [self.crop_size[1], 0.0]], dtype="float32")

        h = cv2.getPerspectiveTransform(src, src+h4pt)

        return patch1,image1, h