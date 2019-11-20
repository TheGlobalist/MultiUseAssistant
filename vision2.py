import cv2
import tensorflow as tf
import numpy as np
import time

def load_model():
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

def detect_objects(image_np, detection_graph, sess):
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

def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
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

            #cv2.circle(image_np, ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), 3, (255, 0, 0))
    return a

def check_movement(sequence, frame, distance=30, y_limit=20):
    sequence = np.array(sequence)
    x = sequence[:, 0]
    y = sequence[:, 1]
    if len(x[x > x[0]]) == FRAMES - 1 and abs(x[0] - x[-1]) >= distance and max(abs(y - y[0])) <= y_limit:
        # mi sto muovendo a sx
        cv2.putText(frame, 'Sinistra', (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    elif len(x[x < x[0]]) == FRAMES - 1 and abs(x[0] - x[-1]) >= distance and max(abs(y - y[0])) <= y_limit:
        # mi sto muovendo a dx
        cv2.putText(frame, 'Destra', (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

detector, sess = load_model()
fist_detector = cv2.CascadeClassifier('models/fist.xml')

threshold = 0.2
num_hands_detect = 2

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
im_width, im_height = (camera.get(3), camera.get(4))

t1 = time.time()
f = 0
c = 0
seq = []
FRAMES = 4

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, scores = detect_objects(frame, detector, sess)

    # draw bounding boxes on frame
    center = draw_box_on_image(num_hands_detect, threshold, scores, boxes, im_width, im_height, frame)

    if center:
        c = 0
        seq.append(center)
        if len(seq) == FRAMES:
            check_movement(seq, frame)
            seq.pop(0)
    else:
        c += 1
        if c == 3:
            seq = []

    # fists detection
    fists = fist_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    for fX, fY, fW, fH in fists:
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    t2 = time.time()
    if t2 - t1 >= 1:
        fps = f
        f = 0
        t1 = time.time()
    else:
        f += 1

    try:
        cv2.putText(frame, 'FPS: ' + str(fps), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    except:
        pass

    cv2.imshow('Hand Detector', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

