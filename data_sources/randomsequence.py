from torch.utils.data import Dataset
import csv
import cv2
import numpy
from math import floor
from random import randint
from math import cos, sin, radians

class RandomSequence():

    def __init__(self, file, translate=10, amount=0, rotation=0, crop_size=(256, 256), patch_size=(128, 128), patch_location = (128, 128)):

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.translate = translate

        self.image = cv2.imread(file)
        #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.amount = amount
        self.rotation = rotation

        self.i = 0

        self.h = 0
        self.v = 0
        self.dir = 0
        self.mov = 0
        self.pos = numpy.array([0,0,0,0,0,0,0,0])

        self.homography = numpy.array([[1, 0, 0],[0,1,0],[0,0,1]], dtype="float32")



    def update_index(self):


        # 0 = RIGHT, 1 = UP, 2 = RIGHT, 3 = DOWN

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

    def update_h(self):
        angle = randint(-self.rotation, self.rotation)

        rotation_matrix = numpy.array([
            [cos(radians(angle)),-sin(radians(angle)), 0],
            [sin(radians(angle)), cos(radians(angle)), 0],
            [0,0,1]], dtype="float32")

        src = numpy.array([[self.center[1] + self.h * self.translate, self.center[0] + self.v * self.translate],
                           [self.center[1] + self.h * self.translate, self.center[0] + self.v * self.translate + self.crop_size[1]],
                           [self.center[1] + self.h * self.translate + self.crop_size[1], self.center[0] + self.v * self.translate + self.crop_size[1]],
                           [self.center[1] + self.h * self.translate + self.crop_size[1], self.center[0] + self.v * self.translate]], dtype="float32")

        diff = numpy.array([[randint(-self.amount, self.amount), randint(-self.amount, self.amount)], \
                                       [randint(-self.amount, self.amount), randint(-self.amount, self.amount)], \
                                       [randint(-self.amount, self.amount), randint(-self.amount, self.amount)], \
                                       [randint(-self.amount, self.amount), randint(-self.amount, self.amount)]],dtype="float32"
                                      )

        self.homography = rotation_matrix

    def get(self, i):

        if (i > self.i):
            self.update_index()
            self.update_h()
            self.i = i

        self.center = (self.image.shape[0]/2, self.image.shape[1]/2)

        image1 = cv2.warpPerspective(self.image, self.homography, (self.image.shape[1], self.image.shape[0]))

        if self.center[0] + self.v * self.translate + self.crop_size[1] > self.image.shape[0] or \
            self.center[0] + self.v * self.translate < 0 or \
            self.center[1] + self.h * self.translate + self.crop_size[1] > self.image.shape[1] or \
            self.center[1] + self.h * self.translate < 0:

            raise StopIteration

        image1 = image1[self.center[0] + self.v * self.translate : self.center[0] + self.v * self.translate + self.crop_size[1],
                 self.center[1] + self.h * self.translate : self.center[1] + self.h * self.translate + self.crop_size[1]]

        patch1 = image1[self.patch_location[0] - self.patch_size[0] / 2:self.patch_location[0] + self.patch_size[0] / 2,
                 self.patch_location[1] - self.patch_size[1] / 2:self.patch_location[1] + self.patch_size[1] / 2]

        patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)

        h4pt = numpy.array(
            [self.h * self.translate, self.v * self.translate, self.h * self.translate, self.v * self.translate,
             self.h * self.translate, self.v * self.translate, self.h * self.translate,
             self.v * self.translate]) - self.pos

        self.pos = numpy.array(
            [self.h * self.translate, self.v * self.translate, self.h * self.translate, self.v * self.translate,
             self.h * self.translate, self.v * self.translate, self.h * self.translate, self.v * self.translate])


        return patch1,image1