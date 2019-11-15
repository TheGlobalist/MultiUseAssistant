from NaturalLanguageUnderstanding import NaturalLanguageUnderstandingModule
from SpeechRecognizer import SpeechRecognizer
from SpeechEngine import SpeechEngine
from Browser import Browser
from GestureRecognition.ContinuousGesturePredictor import GestureRecognitor
from GestureRecognition.DNNGesture import DNN4GestureRecognition
import cv2

NLU = NaturalLanguageUnderstandingModule()
sr = SpeechRecognizer()
speech_engine = SpeechEngine()
browser = Browser()
open_cv_wrapper = GestureRecognitor()
NN = DNN4GestureRecognition()

while open_cv_wrapper.num_frames < 30:
    ########################## OPENCV PART ##########################################

    # Catturo un'istantanea dalla webcam
    (grabbed, frame) = open_cv_wrapper.get_camera().read()
    # Faccio resize dell'immagine
    frame = open_cv_wrapper.resizer(frame)

    # Flip
    frame = open_cv_wrapper.flip(frame)

    clone = frame.copy()

    # Prendo altezza e larghezza
    (height, width) = frame.shape[:2]
    # Le zone della mia region of interest
    top, bottom, right, left = open_cv_wrapper.top, open_cv_wrapper.bottom, open_cv_wrapper.right, open_cv_wrapper.left
    # me la mappo...
    roi = frame[top:bottom, right:left]
    # ...la converto in scala di grigi
    gray = open_cv_wrapper.converter_and_gaussian(roi)
    open_cv_wrapper.run_avg(gray, open_cv_wrapper.aWeight)
    open_cv_wrapper.increase_num_frames()

speech_engine.say("Ciao! Benvenuto. Se hai dei dubbi su come funziono, pronuncia: 'Aiuto'!")
while True:
    sentence = sr.recognize()
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

    #Prendo altezza e larghezza
    (height, width) = frame.shape[:2]
    #Le zone della mia region of interest
    top,bottom,right,left = open_cv_wrapper.top, open_cv_wrapper.bottom,open_cv_wrapper.right, open_cv_wrapper.left
    #me la mappo...
    roi = frame[top:bottom, right:left]
    #...la converto in scala di grigi
    gray = open_cv_wrapper.converter_and_gaussian(roi)
    mano = open_cv_wrapper.segment(gray)
    if mano is not None:
        # if yes, unpack the thresholded image and
        # segmented region
        (thresholded, segmented) = mano
        cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
        open_cv_wrapper.write_and_resize(thresholded)
        norm_image = open_cv_wrapper.prepare_4_prediction()

        predictedClass, confidence = NN.predict(norm_image)
        print("Ho predetto " + str(predictedClass) + " con la confidenza del: " + str(confidence))
    open_cv_wrapper.increase_num_frames()