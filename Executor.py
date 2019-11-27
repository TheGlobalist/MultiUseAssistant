from NaturalLanguageUnderstanding import NaturalLanguageUnderstandingModule
from SpeechRecognizer import SpeechRecognizer
from SpeechEngine import SpeechEngine
from Browser import Browser
from GestureRecognition.ContinuousGesturePredictor import GestureRecognitor
import cv2
import time
import requests
import json

NLU = NaturalLanguageUnderstandingModule()
sr = SpeechRecognizer()
speech_engine = SpeechEngine()
browser = Browser()
open_cv_wrapper = GestureRecognitor()

detector, sess = open_cv_wrapper.load_model()
fist_detector = cv2.CascadeClassifier('models/fist.xml')

threshold = 0.2
num_hands_detect = 1

camera = cv2.VideoCapture(0)
im_width, im_height = (camera.get(3), camera.get(4))
fist_counter = 0
c = 0
seq = []
speech_engine.say(
    "Buongiorno e benvenuto in emGimBot! Prossimamente potrai anche chiedermi come funziono con 'Aiuto'. Intanto, chiedimi una canzone da riprodurre")
while True:
    if not browser.is_active():
        sentence = sr.recognize()
        if not sentence == "":
            dizionario_confidenza = NLU.predictIntention(sentence)
            if dizionario_confidenza['intento'] == 'musica':
                sentence = NLU.tag_sentence(sentence)
                sentence = NLU.get_possible_tags_to_query_uri(sentence)
                browser.navigate_music(sentence)
            if dizionario_confidenza['intento'] == 'aiuto':
                speech_engine.say(
                    "Sono un assistente vocale a cui puoi chiedere le seguenti cose: musica da riprodurre, le ultime notizie, il meteo e come funziono! Inoltre supporto anche le gesture! Sono forte, no?")
                continue
            if dizionario_confidenza['intento'] == 'notizie':
                speech_engine.set_speed(175)  # Meglio rallentarlo
                notizie = requests.get(
                    'https://newsapi.org/v2/top-headlines?country=it&apiKey=f896205045cc40cb947d864b5e2df8f9')
                notizie = json.loads(notizie.text)['articles'][0:5]
                speech_engine.say("Ti leggo i titoli delle ultime 5 notizie!")
                for notizia in notizie:
                    speech_engine.say("Fonte: " + notizia['source']['name'])
                    time.sleep(1)
                    speech_engine.say(notizia['title'])
                speech_engine.set_speed(200)
                continue

    ########################## OPENCV PART ##########################################

    # Catturo un'istantanea dalla webcam
    (grabbed, frame) = camera.read()
    # Faccio resize dell'immagine
    frame = open_cv_wrapper.resizer(frame)

    # Flip
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
            print(movimento)
            browser.movement_detection(movimento)
            seq.pop(0)
        else:
            c += 1
            if c == 3:
                seq = []

    # fists detection
    fists = fist_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
    print(fists, type(fists))
    if not type(fists) == tuple and fists.any():
        fist_counter += 1
    else:
        fist_counter = 0
    if fist_counter >= 5:
        print("PUGNO")
        is_browser_active = browser.is_active()
        if is_browser_active is not None and is_browser_active:
            browser.close()

        pass
camera.release()
cv2.destroyAllWindows()
