from helper.warper import ImageMap
import cv2, numpy
from data_sources.randomsequence import RandomSequence
from data_sources.csvsequence import CsvSequence
from helper import homography_func as hf
from data_sources.videofile import VideoFile
import time

sift = hf.SIFT()

homographynet = hf.HomographyNet("../checkpoints/mild-vgg13-su.pth.tar")
video = CsvSequence("../datasets/example1.csv", base_path="../")

map_size = (1024,1000)
patch_location = (128,128)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(0,500), func=homographynet, data=video, memory=1, distance=100, borders=False, indicate_centers=True) # homography func

time1 = 0
sift_duration = []


homographynet_duration = []

homographynet_time = 0
i = 0
while not imagemap.finished():

    time1 = time.time()

    image = imagemap.warp()

    cv2.imshow("map", image)
    cv2.waitKey(1)

cv2.waitKey(0)