from helper.warper import ImageMap
import cv2, numpy
from data_sources.randomsequence import RandomSequence
from data_sources.csvsequence import CsvSequence
from helper import homography_func as hf

homographynet = hf.HomographyNet("../checkpoints/realdata2-vgg13.pth.tar")
sequence = RandomSequence("../images/19.jpg", translate=5, amount=0, rotation=10)
video = CsvSequence("../datasets/example4.csv", base_path="../")

map_size = (1024,1000)
patch_location = (128,128)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(400,500), func=homographynet, data=video, memory=1, distance=100, borders=False, indicate_centers=True) # homography func

i = 0
while not imagemap.finished():

    image = imagemap.warp()

    cv2.imshow("map", image)
    cv2.waitKey(1)

    if stream1.duration is not None:
        print stream1.duration
    # if stream2.duration is not None:
    #     sift_duration.append(stream2.duration)

    # last_matches = sift.draw_keypoints()
    # if last_matches is not None:
    #     cv2.imshow("matches", last_matches)
    #     cv2.waitKey(1)
    i = i+1
    print i

cv2.waitKey(0)