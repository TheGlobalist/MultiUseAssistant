from NaturalLanguageUnderstanding import NaturalLanguageUnderstandingModule
from SpeechRecognizer import SpeechRecognizer
from SpeechEngine import SpeechEngine
from Browser import Browser
from GestureRecognition.ContinuousGesturePredictor import GestureRecognitor
from GestureRecognition.DNNGesture import DNN4GestureRecognition
import cv2
import os
import io
from PIL import Image
from google.cloud import vision
from google.cloud.vision import types

NLU = NaturalLanguageUnderstandingModule()
sr = SpeechRecognizer()
speech_engine = SpeechEngine()
browser = Browser()
open_cv_wrapper = GestureRecognitor()
NN = DNN4GestureRecognition()
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="GKey.json"
#client = vision.ImageAnnotatorClient()
#stopper = 0

detector, sess = open_cv_wrapper.load_model()
fist_detector = cv2.CascadeClassifier('models/fist.xml')

threshold = 0.2
num_hands_detect = 1

camera = cv2.VideoCapture(0)
im_width, im_height = (camera.get(3), camera.get(4))

"""
while open_cv_wrapper.num_frames < 30:
    ########################## OPENCV PART ##########################################

    # Catturo un'istantanea dalla webcam
    (grabbed, frame) = open_cv_wrapper.get_camera().read()
    # Faccio resize dell'immagine
    frame = open_cv_wrapper.resizer(frame)

    # Flip
    frame = open_cv_wrapper.flip(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    coppia_punti = []
    boxes, scores = open_cv_wrapper.detect_objects(frame, detector, sess)
    fists = None
    if scores[0] < 0.2:
        # fists detection
        fists = fist_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5,
                                               minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

    coppia_punti = open_cv_wrapper.points_for_roi(boxes, scores, im_width, im_height)
    try:
        roi = frame[
              coppia_punti[0][0][1]:coppia_punti[0][1][1],
              coppia_punti[0][0][0]:coppia_punti[0][1][0]
              ]
    except IndexError:
        continue
    # success, encoded_image = cv2.imencode('.png', frame)

    # image = types.Image(content=encoded_image.tobytes())
    clone = frame.copy()

    # Prendo altezza e larghezza

    gray = open_cv_wrapper.converter_and_gaussian(roi)
    open_cv_wrapper.run_avg(gray, im_height)
    open_cv_wrapper.increase_num_frames()
"""
fist_counter = 0
c = 0
seq = []
speech_engine.say("Buongiorno e benvenuto in emGimBot! Prossimamente potrai anche chiedermi come funziono con 'Aiuto'. Intanto, chiedimi una canzone da riprodurre")
while True:
    #sentence = sr.recognize()
    sentence = ""
    if not sentence == "":
        dizionario_confidenza = NLU.predictIntention(sentence)
        #Serie di if-elif per decidere come gestire le operazioni da fargli fare in base a quello che si dice
        if dizionario_confidenza['intento'] == 'musica':
            browser.navigate_music(sentence)
    ########################## OPENCV PART ##########################################

    #Catturo un'istantanea dalla webcam
    (grabbed, frame) = open_cv_wrapper.get_camera().read()
    #Faccio resize dell'immagine
    frame = open_cv_wrapper.resizer(frame)

    #Flip
    frame = open_cv_wrapper.flip(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    coppia_punti = []
    boxes, scores = open_cv_wrapper.detect_objects(frame, detector, sess)
    # draw bounding boxes on frame
    center = open_cv_wrapper.draw_box_on_image(num_hands_detect, threshold, scores, boxes, im_width, im_height, frame)
    if center:
        c = 0
        seq.append(center)
        if len(seq) == 5:
            movimento = open_cv_wrapper.check_movement(seq, frame)
            #print(movimento)
            #TODO Logica da usare per il movimento
            seq.pop(0)
        else:
            c += 1
            if c == 3:
                seq = []

    # fists detection
    fists = fist_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
    #print(fists, type(fists))
    if not type(fists) == tuple and fists.any():
        fist_counter += 1
    else:
        fist_counter = 0
    if fist_counter >= 5:
        #print("PUGNO")
        #TODO Detection per chiudere il browser
        pass
    cv2.imshow('Hand Detector', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
camera.release()
cv2.destroyAllWindows()
