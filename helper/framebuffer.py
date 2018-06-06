from threading import Thread
import sys
import cv2

from Queue import Queue


class FrameBuffer:
    def __init__(self, path, start, advance, queueSize=128):

        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.pos = start
        self.advance = advance
        self.t = Thread(target=self.update, args=())
        self.Q = Queue(maxsize=queueSize)

    def start(self):

        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while True:

            if self.stopped:
                return

            if not self.Q.full():

                self.stream.set(1, self.pos)
                self.pos += self.advance

                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stop()
                    return

                self.Q.put(frame)

    def read(self):
        if not self.t.isAlive():
            return False
        return self.Q.get()