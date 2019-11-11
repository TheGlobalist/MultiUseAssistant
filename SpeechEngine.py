import pyttsx3

class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()

    def set_speed(self, speed):
        self.engine.setProperty('rate',speed)

    def say(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

