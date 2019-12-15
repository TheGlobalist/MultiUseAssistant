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

threshold = 0.2
num_hands_detect = 2
f = 0
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
im_width, im_height = (camera.get(3), camera.get(4))
fist_counter = 0
c = 0
s = 0
t1 = time.time()
seq = []
speech_engine.set_speed(175)
speech_engine.say("Benvenuto in emGiBot!")
while True:
    if not browser.is_active():
        #sentence = sr.recognize()
        sentence = "Riproduci Orphans dei Coldplay"
        if not sentence == "":
            dizionario_confidenza = NLU.predictIntention(sentence)
            if dizionario_confidenza['intento'] == 'musica':
                sentence = NLU.format_query_uri(sentence)
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
                    time.sleep(1)
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
        if len(center) == 1:
            s = 0
            c = 0
            seq.append(center[0])
            if len(seq) == 5:
                movimento = open_cv_wrapper.check_movement(seq, frame)
                print(movimento)
                browser.movement_detection(movimento)
                seq.pop(0)
        else:
            s += 1
            if s >= 5:
                is_browser_active = browser.is_active()
                if is_browser_active is not None and is_browser_active:
                    browser.close()
                    s = 0
                    continue
    else:
        c += 1
        if c == 3:
            seq = []
            s = 0


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
