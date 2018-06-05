from helper.warper import ImageMap
import cv2, numpy
from data_sources.randomsequence import RandomSequence
from data_sources.videofile import VideoFile
from helper import homography_func as hf
from data_sources.videofile import VideoFile
import time


homographynet = hf.HomographyNet("../checkpoints/realdata2-vgg13.pth.tar")
video = VideoFile("../video/5.mp4", advance=2, start=4240, crop_size=(1000,1000))

map_size = (1024,1000)
patch_location = (128,128)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(600,0), func=homographynet, data=video, memory=1, distance=100, borders=False, indicate_centers=True) # homography func

while not imagemap.finished():

    image = imagemap.warp()

    cv2.imshow("map", image)
    cv2.waitKey(1)

cv2.waitKey(0)