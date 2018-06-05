from netinstance import NetInstance
import sys
import numpy, cv2, torch

class HomographyNet:
    def __init__(self, checkpoint_filename):
        self.netinstance = NetInstance(input_file=checkpoint_filename)

    def __call__(self, images1, image2, warp_images1, warp_image2):

        input = torch.stack([torch.stack([torch.from_numpy(image1), torch.from_numpy(image2)]) for image1 in images1])
        image = torch.stack([torch.from_numpy(warp_image1) for warp_image1 in warp_images1])

        _,_,all_h4pt = self.netinstance.evaluate(input, image)

        all_h4pt = all_h4pt.cpu().data.numpy()

        image_width_c, image_height_c = image2.shape[0] / 2, image2.shape[1] / 2
        warp_image_width_c, warp_image_height_c = warp_image2.shape[0] / 2, warp_image2.shape[1] / 2

        h = []
        for h4pt in numpy.split(all_h4pt,len(images1)):
            h4pt = h4pt.squeeze(axis=0)
            src = numpy.array([[image_width_c-warp_image_width_c, image_height_c-warp_image_height_c],
                               [image_width_c-warp_image_width_c, image_height_c+warp_image_height_c],
                               [image_width_c+warp_image_width_c, image_height_c+warp_image_height_c],
                               [image_width_c+warp_image_width_c, image_height_c-warp_image_height_c]], dtype="float32")

            dst = src + numpy.array([[h4pt[0], h4pt[1]],
                                     [h4pt[2], h4pt[3]],
                                     [h4pt[4], h4pt[5]],
                                     [h4pt[6], h4pt[7]]], dtype="float32")

            h.append(cv2.getPerspectiveTransform(src, dst))

        return h


class SIFT:

    def __call__(self, images1, image2, warp_images1, warp_image2):

        h_seq = []
        for image1, warp_image1 in zip(images1, warp_images1):

            self.warp_image2 = warp_image2
            self.warp_image1 = warp_image1

            descriptor = cv2.xfeatures2d.SIFT_create()
            (kpsA, featuresA) = descriptor.detectAndCompute(warp_image2, None)

            self.kpsA = numpy.float32([kp.pt for kp in kpsA])

            descriptor = cv2.xfeatures2d.SIFT_create()
            (kpsB, featuresB) = descriptor.detectAndCompute(warp_image1, None)

            self.kpsB = numpy.float32([kp.pt for kp in kpsB])

            reprojThresh = 4.0
            ratio = 0.75

            matcher = cv2.DescriptorMatcher_create("BruteForce")
            self.matches = []
            if featuresA is not None and featuresB is not None:
                rawMatches = matcher.knnMatch(featuresA, featuresB, 2)


                for m in rawMatches:
                    if (len(m) == 2) and m[0].distance < m[1].distance*ratio:
                        self.matches.append((m[0].trainIdx, m[0].queryIdx))


            if len(self.matches) > 4:

                ptsA = numpy.float32([self.kpsA[i] for (_, i) in self.matches])
                ptsB = numpy.float32([self.kpsB[i] for (i, _) in self.matches])


                (H, self.status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                                 reprojThresh)

                h_seq.append(H)
            else:
                self.status = [False for i in self.matches]



        return h_seq


    def draw_last_matches(self):
        if hasattr(self, 'warp_image1'):

            (hA, wA) = self.warp_image1.shape[:2]
            (hB, wB) = self.warp_image2.shape[:2]
            vis = numpy.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
            vis[0:hA, 0:wA] = self.warp_image1
            vis[0:hB, wA:] = self.warp_image2


            for ((trainIdx, queryIdx), s) in zip(self.matches, self.status):

                if s == 1:

                    ptA = (int(self.kpsA[queryIdx][0]), int(self.kpsA[queryIdx][1]))
                    ptB = (int(self.kpsB[trainIdx][0]) + wA, int(self.kpsB[trainIdx][1]))
                    cv2.line(vis, ptA, ptB, (0, 255, 0), 1)


            return vis

        else:
            return None

    def draw_keypoints(self):
        if hasattr(self, 'warp_image2'):

            (hB, wB) = self.warp_image2.shape[:2]
            vis = numpy.zeros((hB, wB, 3), dtype="uint8")
            vis[0:hB, :] = self.warp_image2

            for ((trainIdx, queryIdx), s) in zip(self.matches, self.status):
                color = (0,255,0) if s else (0,0,255)
                ptB = (int(self.kpsB[trainIdx][0]), int(self.kpsB[trainIdx][1]))
                cv2.circle(vis, ptB, 5, color, 1)


            return vis

        else:
            return None

class SURF:

    def __call__(self, images1, image2, warp_images1, warp_image2):

        h_seq = []
        for image1, warp_image1 in zip(images1, warp_images1):

            self.warp_image2 = warp_image2
            self.warp_image1 = warp_image1

            descriptor = cv2.xfeatures2d.SURF_create()
            (kpsA, featuresA) = descriptor.detectAndCompute(warp_image2, None)

            self.kpsA = numpy.float32([kp.pt for kp in kpsA])

            descriptor = cv2.xfeatures2d.SURF_create()
            (kpsB, featuresB) = descriptor.detectAndCompute(warp_image1, None)

            self.kpsB = numpy.float32([kp.pt for kp in kpsB])

            reprojThresh = 4.0
            ratio = 0.75

            matcher = cv2.DescriptorMatcher_create("BruteForce")
            self.matches = []
            if featuresA is not None and featuresB is not None:
                rawMatches = matcher.knnMatch(featuresA, featuresB, 2)


                for m in rawMatches:
                    if (len(m) == 2) and m[0].distance < m[1].distance*ratio:
                        self.matches.append((m[0].trainIdx, m[0].queryIdx))

            if len(self.matches) > 4:

                ptsA = numpy.float32([self.kpsA[i] for (_, i) in self.matches])
                ptsB = numpy.float32([self.kpsB[i] for (i, _) in self.matches])

                (H, self.status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                                 reprojThresh)


                h_seq.append(H)
            else:
                self.status = [False for i in self.matches]


        return h_seq



    def draw_last_matches(self):
        if hasattr(self, 'warp_image1'):

            (hA, wA) = self.warp_image1.shape[:2]
            (hB, wB) = self.warp_image2.shape[:2]
            vis = numpy.zeros((max(hA, hB), wA + wB+50, 3), dtype="uint8")
            vis.fill(255)
            vis[0:hA, 0:wA] = self.warp_image1
            vis[0:hB, 50+wA:] = self.warp_image2

            for ((trainIdx, queryIdx), s) in zip(self.matches, self.status):

                if s == 1:

                    ptA = (int(self.kpsA[queryIdx][0]), int(self.kpsA[queryIdx][1]))
                    ptB = (int(self.kpsB[trainIdx][0]) + wA+50, int(self.kpsB[trainIdx][1]))
                    cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

            return vis

        else:
            return None

    def draw_keypoints(self):
        if hasattr(self, 'warp_image2'):

            (hB, wB) = self.warp_image2.shape[:2]
            if self.warp_image2.ndim == 3:
                vis = numpy.zeros((hB, wB, 3), dtype="uint8")
            else:
                vis = numpy.zeros((hB, wB), dtype="uint8")
            vis[0:hB, :] = self.warp_image2

            for ((trainIdx, queryIdx), s) in zip(self.matches, self.status):
                color = (0,255,0) if s else (0,0,255)
                ptB = (int(self.kpsB[trainIdx][0]), int(self.kpsB[trainIdx][1]))
                cv2.circle(vis, ptB, 5, color, 1)

            return vis

        else:
            return None
