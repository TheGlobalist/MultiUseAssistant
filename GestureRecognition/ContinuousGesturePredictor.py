import numpy as np
from PIL import Image
import cv2
from .DNNGesture import DNN4GestureRecognition
import imutils

class GestureRecognitor:
    def __init__(self):
        self.aWeight = 0.5
        self.__camera = cv2.VideoCapture(0)
        # region of interest (ROI) coordinates
        self.top, self.right, self.bottom, self.left = 10, 350, 225, 590
        # initialize num of frames
        self.num_frames = 0
        self.__bg = None

    def increase_num_frames(self):
        self.num_frames += 1

    def get_camera(self):
        return self.__camera

    def __load_neural_network(self):
        self.__network = DNN4GestureRecognition()

    def resizer(self, frame):
        return imutils.resize(frame, width=700)

    def flip(self, frame):
        return cv2.flip(frame,1)

    def converter_and_gaussian(self,roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (7, 7), 10, sigmaY=10)
        return gray

    def run_avg(self,image, aWeight):
        # initialize the background
        if self.__bg is None:
            self.__bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.__bg, aWeight)

    def segment(self, image, threshold=55):
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.__bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff,
                                    threshold,
                                    255,
                                    cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        (cnts, _) = cv2.findContours(thresholded.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    def __resizeImage(self,imageName):
        basewidth = 100
        img = Image.open(imageName)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(imageName)

    def write_and_resize(self, image):
        cv2.imwrite('Temp.png', image)
        self.__resizeImage('Temp.png')

    def prepare_4_prediction(self):
        # Predict
        image = cv2.imread('Temp.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.normalize(gray_image, gray_image, 0, 255, cv2.NORM_MINMAX)
        return gray_image

