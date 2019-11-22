import numpy as np
from PIL import Image
import cv2
from .DNNGesture import DNN4GestureRecognition
import imutils
import tensorflow as tf

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

    def load_model(self):
        print('Loading hand detector...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('models/model.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
        print("Hand detector loaded.")
        return detection_graph, sess

    def detect_objects(self,image_np, detection_graph, sess):
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name(
            'num_detections:0')

        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores,
             detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        return np.squeeze(boxes), np.squeeze(scores)

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
        self.__bg = gray.copy().astype("float")
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

    def check_movement(self,sequence, frame, distance=30, y_limit=20):
        sequence = np.array(sequence)
        x = sequence[:, 0]
        y = sequence[:, 1]
        if len(x[x > x[0]]) == 4 and abs(x[0] - x[-1]) >= distance and max(abs(y - y[0])) <= y_limit:
            # mi sto muovendo a sx
            return "SX"
        elif len(x[x < x[0]]) == 4 and abs(x[0] - x[-1]) >= distance and max(abs(y - y[0])) <= y_limit:
            # mi sto muovendo a dx
            return "DX"
        return None

    def draw_box_on_image(self,num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
        a = []
        k = 0
        for i in range(num_hands_detect):
            if (scores[i] > score_thresh):
                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                              boxes[i][0] * im_height, boxes[i][2] * im_height)
                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))
                cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
                if k == 0:
                    a.append((p1[0] + p2[0]) // 2)
                    a.append((p1[1] + p2[1]) // 2)
                k += 1

                # cv2.circle(image_np, ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), 3, (255, 0, 0))
        return a
