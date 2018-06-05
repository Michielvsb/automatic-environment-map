from helper.warper import ImageMap
import cv2
from helper import homography_func as hf
from data_sources.squarewavesequence import SquareWaveSequence

sift = hf.SIFT()
surf = hf.SURF()

homographynet = hf.HomographyNet("../checkpoints/mild-vgg13-su.pth.tar")
sequence = SquareWaveSequence("../images/19.jpg", translate=5, amount=0)

map_size = (900,1000)

imagemap = ImageMap(map_size)

stream1 = imagemap.create_stream(position=(200,0), func=homographynet, data=sequence, memory=2, distance=100, indicate_centers=True, last_border=True)
stream2 = imagemap.create_stream(position=(600,0), func=surf, data=sequence, memory=1, distance=100, indicate_centers=True)

map = imagemap.warp()

cv2.imshow("map", map)
cv2.waitKey(1)

while not imagemap.finished():

    map = imagemap.warp()

    cv2.imshow("map", map)
    cv2.waitKey(1)

    last_matches = surf.draw_keypoints()
    if last_matches is not None:
        cv2.imshow("surf", last_matches)
        cv2.waitKey(1)

    last_matches = sift.draw_keypoints()
    if last_matches is not None:
        cv2.imshow("sift", last_matches)
        cv2.waitKey(1)

    last_image = stream1.last_image
    if last_image is not None:
        cv2.imshow("last_image", last_image)
        cv2.waitKey(1)

    last_patch = stream1.last_patch
    if last_patch is not None:
        cv2.imshow("last_patch", last_patch)
        cv2.waitKey(0)

cv2.waitKey(0)