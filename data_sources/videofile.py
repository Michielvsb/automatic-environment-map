import cv2
from helper.framebuffer import FrameBuffer

patch_size=(128, 128)
patch_location = (128, 128)

class VideoFile():

    def __init__(self, file, advance=5, start=0, crop_size=(600,600)):
        self.file = file
        self.framebuffer = FrameBuffer("{}".format(self.file), start, advance)
        self.framebuffer.start()
        self.current_frame_nb = -1
        self.current_frame = False
        self.crop_size = crop_size

    def get(self, i):

        if self.current_frame_nb != i:
            self.current_frame = self.framebuffer.read()
            if self.current_frame is not False:
                self.current_frame_nb = i
            else:
                raise StopIteration

        width, height = self.current_frame.shape[1], self.current_frame.shape[0]



        color_image1 = self.current_frame[height/2-self.crop_size[0]/2:height/2+self.crop_size[0]/2, width/2-self.crop_size[1]/2:width/2+self.crop_size[1]/2]
        color_image1 = cv2.resize(color_image1, (256, 256), interpolation=cv2.INTER_AREA)
        image1 = cv2.cvtColor(color_image1, cv2.COLOR_BGR2GRAY)
        patch1 = image1[patch_location[0] - patch_size[0] / 2:patch_location[0] + patch_size[0] / 2,
                 patch_location[1] - patch_size[1] / 2:patch_location[1] + patch_size[1] / 2]

        return patch1, color_image1, None
