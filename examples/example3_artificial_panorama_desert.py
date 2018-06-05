from helper.warper import ImageMap
import cv2
from helper import homography_func as hf
from data_sources.squarewavesequence import SquareWaveSequence

sift = hf.SIFT()
surf = hf.SURF()
homographynet = hf.HomographyNet("../checkpoints/mild-vgg13-su.pth.tar")
sequence = SquareWaveSequence("../images/15.jpg", translate=4, amount=0)

map_size = (800,1200)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(200,0), func=homographynet, data=sequence, memory=1, distance=100, borders=False, indicate_centers=True, last_border=True) # homography func
stream2 = imagemap.create_stream(position=(600,0), func=surf, data=sequence, memory=1, distance=100, indicate_centers=True) # sift

while not imagemap.finished():

    map = imagemap.warp()

    cv2.imshow("map", map)
    cv2.waitKey(1)

    last_matches = surf.draw_keypoints()
    if last_matches is not None:
         cv2.imshow("SURF Keypoints", last_matches)
         cv2.waitKey(1)

cv2.waitKey(0)