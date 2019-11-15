import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self):
        self.__recognizer = sr.Recognizer()

    def recognize(self):
        with sr.Microphone() as source:
            audio = self.__recognizer.listen(source)
        recognized_sentence = self.__recognizer.recognize_google(audio, language="it-IT")
        return recognized_sentence
