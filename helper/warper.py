import numpy, cv2, copy, time

class ImageMap:

    def __init__(self, size):
        self.height, self.width = size
        self.map = numpy.zeros((self.height, self.width,3), dtype="uint8")
        self.map.fill(255)
        self.streams = []

    def create_stream(self, **kwargs):

        stream = ImageStream(map=self, **kwargs)
        self.streams.append(stream)

        return stream

    def draw(self, image, h):

        width = self.map.shape[1]
        height = self.map.shape[0]

        mask = numpy.zeros((image.shape[0], image.shape[1]), dtype="uint8")
        mask[:] = 255
        mask_warp = cv2.warpPerspective(mask, h, (width, height))
        image_warp = cv2.warpPerspective(image, h, (width, height))

        locs = numpy.where(mask_warp == 255)
        self.map[locs[0], locs[1]] = image_warp[locs[0], locs[1]]

    def warp(self):
        map = copy.deepcopy(self.map)

        for stream in list(self.streams):
            try:
                centers, ground_truth_centers, last_image = stream.warp()

                map = self.draw_centers(map, centers, (255,0,0))
                map = self.draw_centers(map, ground_truth_centers, (0,255,0))

                if stream.last_border and last_image is not None:
                    last_image.estimate_borders().draw(map, (255, 255, 255))

            except StopIteration:
                self.streams.remove(stream)
                map = self.draw_centers(map, stream.warped_centers, (255,0,0))
                map = self.draw_centers(map, stream.ground_truth_centers, (0,255,0))

        return map

    def finished(self):

        return len(self.streams) == 0

    def clear(self):
        self.map = numpy.zeros((self.height, self.width, 3), dtype="uint8")

    def image(self):
        return self.map

    def draw_centers(self, map, centers, color):
        previous_center = None
        for center in centers:
            if previous_center is not None:
                map = cv2.line(map, previous_center, center, color)
            map = cv2.circle(map, center, 3, (255, 255, 255), 1)
            previous_center = center
        return map

class LRUList:

    def __init__(self, capacity):
        self.capacity=capacity
        self.items = []
        self.i = -1

    def add(self, item):
        if len(self.items) == self.capacity:
            self.updateCnt()
            self.items[self.i] = item

        else:
            self.updateCnt()
            self.items.append(item)

    def empty(self):
        return len(self.items) == 0

    def getLatest(self):
        return self.items[self.i]

    def getSecondLatest(self):
        if self.i-1 == -1:
            return self.items[self.capacity-1]
        else:
            return self.items[self.i-1]

    def updateCnt(self):
        if self.i == self.capacity-1:
            self.i = 0
        else:
            self.i += 1

    def __iter__(self):
        self.currentItem = 0
        return self

    def next(self):
        if self.currentItem == self.capacity or self.currentItem >= len(self.items):
            raise StopIteration
        else:
            item = self.items[self.currentItem]
            self.currentItem += 1
            return item

    def images(self):
        h = [item.H for item in self.items]
        images = [item.image for item in self.items]
        warp_images = [item.warp_image for item in self.items]

        return h, images, warp_images

    def __len__(self):
        return len(self.items)


class Border:

    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def __sub__(self, other):

        diff = numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(other.p1, self.p1), 2)))
        diff += numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(other.p2, self.p2), 2)))
        diff += numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(other.p3, self.p3), 2)))
        diff += numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(other.p4, self.p4), 2)))

        return diff/4

    @staticmethod
    def avg_border_distance(borders):

        if len(borders) < 2:
            return 0

        ref_border = borders[0];
        total_border_distance = 0
        for border in borders[1:]:
            total_border_distance += ref_border - border;

        return total_border_distance / (len(borders) -1);

    def rectangle(self):

        x1, y1 = self.p1
        x2, y2 = self.p2
        x3, y3 = self.p3
        x4, y4 = self.p4

        return Border((max(x1,x2),max(y1,y4)),(max(x1,x2),min(y2,y3)),(min(x3,x4),min(y2,y3)),(min(x3,x4),max(y1,y4)))

    def draw(self, image, color):

        cv2.line(image, self.p1, self.p2, color)
        cv2.line(image, self.p2, self.p3, color)
        cv2.line(image, self.p3, self.p4, color)
        cv2.line(image, self.p4, self.p1, color)

        return image

class MatchedImage:

    def __init__(self, image, warp_image, H):
        self.image = image
        self.warp_image = warp_image
        self.H = H

    def estimate_borders(self):

        width = self.warp_image.shape[1]
        height = self.warp_image.shape[0]

        m1 = numpy.dot(numpy.array([0,0,1]), cv2.transpose(self.H))
        m2 = numpy.dot(numpy.array([0,height,1]), cv2.transpose(self.H))
        m3 = numpy.dot(numpy.array([width,height,1]), cv2.transpose(self.H))
        m4 = numpy.dot(numpy.array([width,0,1]), cv2.transpose(self.H))

        p1 = tuple(numpy.rint(numpy.divide(m1[0:2], m1[2])).astype(int))
        p2 = tuple(numpy.rint(numpy.divide(m2[0:2], m2[2])).astype(int)[0:2])
        p3 = tuple(numpy.rint(numpy.divide(m3[0:2], m3[2])).astype(int)[0:2])
        p4 = tuple(numpy.rint(numpy.divide(m4[0:2], m4[2])).astype(int)[0:2])

        return Border(p1,p2,p3,p4)

class ImageStream:
    def __init__(self, map, position, func, data, memory=1, distance=10, borders=False, indicate_centers=False, last_border=False):
        self.position = position
        self.map = map
        self.last_images = LRUList(memory)
        self.distance = distance
        self.image_cnt = 0
        self.homography_func = func
        self.data = data
        self.data_position = 0
        self.borders=borders
        self.duration = None
        self.warped_centers = []
        self.ground_truth_centers = []
        self.indicate_centers = indicate_centers
        self.ground_truth_h = None
        self.last_border = last_border
        self.last_image = None
        self.last_patch = None


    def warp(self):
        duration = 0

        image, warp_image, ground_truth_h = self.data.get(self.data_position)
        self.last_image = warp_image
        self.last_patch = image

        center = numpy.array([[warp_image.shape[1]/2],[warp_image.shape[0]/2],[1.0]], dtype="float32")
        self.data_position += 1

        matched_image = None
        h_avg = None
        if self.last_images.empty():
            transl_h = self.position[0]
            transl_w = self.position[1]

            h_avg = numpy.array([[1, 0, transl_w], [0, 1, transl_h], [0, 0, 1]], dtype="float")

            self.map.draw(warp_image, h_avg)

            matched_image = MatchedImage(image, warp_image, h_avg)
            self.last_images.add(matched_image)

            if self.indicate_centers:
                self.warped_centers.append(self.warp_point(h_avg, center))

                if ground_truth_h is not None:

                    self.ground_truth_h = numpy.matmul(ground_truth_h,h_avg)
                    self.ground_truth_centers.append(self.warp_point(self.ground_truth_h, center))

            self.duration = None

        else:
            h_acc = numpy.zeros((3, 3))
            all_borders = []

            last_hs, last_images, last_warp_images = self.last_images.images()

            start = time.time()
            h = self.homography_func(last_images, image, last_warp_images, warp_image)
            duration = time.time() - start

            self.duration = duration

            h_acc = []
            for last_h, last_image, last_warp_image, h in zip(last_hs, last_images, last_warp_images, h):

                if h is not None:
                    h_acc.append(numpy.matmul(last_h, h))
                    current_image = MatchedImage(image, warp_image, h)
                    all_borders.append(current_image.estimate_borders())



            border_distance = Border.avg_border_distance(all_borders)


            if len(h_acc) > 0 and border_distance < self.distance:
                h_avg = numpy.divide(sum(h_acc), len(h_acc))

                matched_image = MatchedImage(image, warp_image, h_avg)


                self.map.draw(warp_image, h_avg)

                if self.borders:
                    matched_image.estimate_borders().draw(self.map.map, (255, 255, 255))

                self.last_images.add(matched_image)

                if self.indicate_centers:
                    self.warped_centers.append(self.warp_point(h_avg, center))

                    if ground_truth_h is not None:
                        self.ground_truth_h = numpy.matmul(ground_truth_h,self.ground_truth_h)
                        self.ground_truth_centers.append(self.warp_point(self.ground_truth_h, center))

        if len(self.last_images) > 1 and h_avg is not None:
            h_return = numpy.matmul(h_avg, numpy.linalg.inv(self.last_images.getSecondLatest().H))
        else:
            h_return = None

        return self.warped_centers, self.ground_truth_centers, matched_image

    def warp_point(self, h, point):
        warped_center = numpy.matmul(h, point)
        warped_center = (warped_center / warped_center[2, 0])
        warped_center = (int(warped_center[0, 0]), int(warped_center[1, 0]))
        return warped_center


    def borders(self, image, warp_image):

        colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255,255,0),(0,255,255),(255,0,255),(255,255,255)]

        map = copy.deepcopy(self.map.map)

        h_acc = numpy.zeros((3, 3))
        all_borders = []

        for i, lastImage in enumerate(self.last_images):
            h = self.homography_func(lastImage.image, image, lastImage.warp_image, warp_image)

            if h is None:
                h_acc = None
                break
            else:
                h = numpy.matmul(lastImage.H, h)
                h_acc += h
                current_image = MatchedImage(image, warp_image, h)
                map = current_image.estimate_borders().draw(map, colors[i])

        return map

    def get_initial_h4pt(self):
        return numpy.array([self.position[1], self.position[0],self.position[1], self.position[0],self.position[1], self.position[0],self.position[1], self.position[0]])