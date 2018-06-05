from torch.utils.data import Dataset
import csv
import cv2
import numpy
from math import floor


class CsvSequence():

    def __init__(self, file, base_path=""):

        with open(file, 'rb') as f:
            reader = csv.reader(f)
            self.data = list(reader)

        self.i = 0
        self.base_path = base_path

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.i = 0
        return self

    def next(self):

        if (self.i >= len(self.data)):
            raise StopIteration

        data_item = self.data[self.i]

        image1 = cv2.imread("datasets/"+data_item[3])
        patch1 = cv2.imread("datasets/"+data_item[1], cv2.IMREAD_GRAYSCALE)

        self.i += 1

        return image1,patch1,None

    def __getitem__(self, item):
        data_item = self.data[item]

        image1 = cv2.imread("datasets/"+data_item[3])
        patch1 = cv2.imread("datasets/"+data_item[1], cv2.IMREAD_GRAYSCALE)

        return image1, patch1, None

    def get(self, i):

        if i >= len(self.data):
            raise StopIteration

        data_item = self.data[i]

        image1 = cv2.imread(self.base_path+"datasets/"+data_item[3])
        patch1 = cv2.imread(self.base_path+"datasets/"+data_item[1], cv2.IMREAD_GRAYSCALE)

        return patch1, image1, None