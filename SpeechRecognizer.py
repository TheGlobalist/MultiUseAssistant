import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self):
        self.__recognizer = sr.Recognizer()

    def recognize(self):
        with sr.Microphone() as source:
            audio = r.listen(source)
        recognized_sentence = r.recognize_google(audio, language="it-IT")
        return recognized_sentence
