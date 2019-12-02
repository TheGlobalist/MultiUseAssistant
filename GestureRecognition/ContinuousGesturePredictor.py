import numpy as np
from PIL import Image
import cv2
from .DNNGesture import DNN4GestureRecognition
import imutils
import tensorflow as tf

class GestureRecognitor:
    def __init__(self):
        pass


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


    def resizer(self, frame):
        return imutils.resize(frame, width=700)

    def flip(self, frame):
        return cv2.flip(frame,1)

    def check_movement(self,sequence, frame, distance=30, y_limit=20):
        sequence = np.array(sequence)
        x = sequence[:, 0]
        y = sequence[:, 1]
        if len(x[x > x[0]]) == 4 and abs(x[0] - x[-1]) >= distance and max(abs(y - y[0])) <= y_limit:
            # mi sto muovendo a sx
            return "DX"
        elif len(x[x < x[0]]) == 4 and abs(x[0] - x[-1]) >= distance and max(abs(y - y[0])) <= y_limit:
            # mi sto muovendo a dx
            return "SX"
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
