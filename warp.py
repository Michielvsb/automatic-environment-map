from helper.warper import ImageMap
import cv2, numpy
from data_sources.randomsequence import RandomSequence
from data_sources.videofile import VideoFile
from helper import homography_func as hf
from data_sources.videofile import VideoFile
import time

surf = hf.SURF()
sift = hf.SIFT()

homographynet = hf.HomographyNet("checkpoints/realdata2-vgg13.pth.tar")
sequence = RandomSequence("images/19.jpg", translate=5, amount=0, rotation=10)
video = VideoFile("video/4.mp4", advance=4, start=7300, crop_size=(1000,1000))

fps = 10

map_size = (1400,1000)
patch_location = (128,128)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(750,200), func=homographynet, data=video, memory=1, distance=100, borders=False, indicate_centers=True, last_border=False) # homography func
stream2 = imagemap.create_stream(position=(600,600), func=surf, data=video, memory=1, distance=100, indicate_centers=True) # sift
#stream3 = imagemap.create_stream(position=(800,600), func=sift, data=video, memory=1, distance=100, indicate_centers=True)

time1 = 0
sift_duration = []


homographynet_duration = []

homographynet_time = 0

image = imagemap.warp()

cv2.imshow("map", image)
cv2.waitKey(1)

image = imagemap.warp()

cv2.imshow("map", image)
cv2.waitKey(1)

last_matches = surf.draw_last_matches()
if last_matches is not None:
    cv2.imshow("surf_matches", last_matches)
    cv2.waitKey(0)

while not imagemap.finished():


    time1 = time.time()

    image = imagemap.warp()


    cv2.imshow("map", image)
    cv2.waitKey(1)

    # if stream1.last_image is not None:
    #     print stream1.duration
    #
    #     cv2.imshow("last_image", stream1.last_image)
    #     cv2.waitKey(1)
    #
    # if stream1.last_patch is not None:
    #     print stream1.duration
    #
    #     cv2.imshow("last_patch", stream1.last_patch)
    #     cv2.waitKey(0)
    # if stream2.duration is not None:
    #     sift_duration.append(stream2.duration)

    last_matches = surf.draw_last_matches()
    if last_matches is not None:
        cv2.imshow("surf_matches", last_matches)
        cv2.waitKey(1)

    # last_matches = sift.draw_last_matches()
    # if last_matches is not None:
    #     cv2.imshow("sift_matches", last_matches)
    #     cv2.waitKey(1)

    # keypoints = surf.draw_keypoints()
    # if keypoints is not None:
    #     cv2.imshow("surf_keypoints", keypoints)
    #     cv2.waitKey(1)

    # keypoints = sift.draw_keypoints()
    # if keypoints is not None:
    #     cv2.imshow("sift_keypoints", keypoints)
    #     cv2.waitKey(0)

homographynet_duration = numpy.array(homographynet_duration)
sift_duration = numpy.array(sift_duration)

print homographynet_duration[3:].mean()
print homographynet_duration[3:].std()
# print sift_duration[3:].mean()
# print sift_duration[3:].std()

cv2.waitKey(0)