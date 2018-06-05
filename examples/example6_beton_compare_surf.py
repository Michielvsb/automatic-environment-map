from helper.warper import ImageMap
import cv2, numpy
from data_sources.csvsequence import CsvSequence
from helper import homography_func as hf
from data_sources.videofile import VideoFile
import time

surf = hf.SURF()

homographynet = hf.HomographyNet("../checkpoints/realdata2-vgg13.pth.tar")
video = CsvSequence("../datasets/example6.csv", base_path="../")

map_size = (1024,1000)
patch_location = (128,128)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(0,200), func=homographynet, data=video, memory=1, distance=100, borders=False, indicate_centers=True) # homography func
stream2 = imagemap.create_stream(position=(0,600), func=surf, data=video, memory=1, distance=100) # sift

while not imagemap.finished():

    image = imagemap.warp()

    cv2.imshow("map", image)
    cv2.waitKey(1)

    last_matches = surf.draw_keypoints()
    if last_matches is not None:
        cv2.imshow("matches", last_matches)
        cv2.waitKey(1)


cv2.waitKey(0)