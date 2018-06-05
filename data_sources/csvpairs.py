from torch.utils.data import Dataset
import csv
import cv2
import numpy
from math import floor

class CsvPairs():

    def __init__(self, file, cnt = 0):

        with open(file, 'rb') as f:
            reader = csv.reader(f)
            self.data = list(reader)

        self.i = 0

        if cnt>0:
            self.data = self.data[0:cnt]

    def __iter__(self):
        self.i = 0
        return self

    def next(self):

        if (self.i >= len(self.data)):
            raise StopIteration

        data_item = self.data[self.i]

        patch1 = cv2.imread("datasets/"+data_item[1], cv2.IMREAD_GRAYSCALE)
        patch2 = cv2.imread("datasets/" + data_item[2], cv2.IMREAD_GRAYSCALE)
        ground_truth = numpy.asarray(map(float, data_item[3:]))

        self.i += 1

        return patch1, patch2, ground_truth