import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self):
        self.__recognizer = sr.Recognizer()

    def recognize(self):
        with sr.Microphone() as source:
            try:
                audio = self.__recognizer.listen(source, timeout=5)
            except sr.WaitTimeoutError:
                return ""
        try:
            recognized_sentence = self.__recognizer.recognize_google(audio, language="it-IT")
        except sr.UnknownValueError:
            return ""
        return recognized_sentence
