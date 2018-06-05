from helper.warper import ImageMap
import cv2
from helper import homography_func as hf
from data_sources.csvsequence import CsvSequence


homographynet = hf.HomographyNet("../checkpoints/realdata2-vgg13.pth.tar")
video = CsvSequence("../datasets/example8.csv", base_path="../")

map_size = (1024,1000)
patch_location = (128,128)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(200,350), func=homographynet, data=video, memory=1, distance=100, borders=False, indicate_centers=True) # homography func

while not imagemap.finished():

    image = imagemap.warp()

    cv2.imshow("map", image)
    cv2.waitKey(1)


cv2.waitKey(0)