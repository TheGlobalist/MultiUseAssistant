from NaturalLanguageUnderstanding import NaturalLanguageUnderstandingModule
from SpeechRecognizer import SpeechRecognizer
from SpeechEngine import SpeechEngine


NLU = NaturalLanguageUnderstandingModule()
sr = SpeechRecognizer()
speech_engine = SpeechEngine()

speech_engine.say("Ciao! Benvenuto. Se hai dei dubbi su come funziono, pronuncia: 'Aiuto'!")
while True:

    sentence = sr.recognize()
    
    dizionario_confidenza = NLU.predictIntention(sentence)
    #Serie di if-elif per decidere come gestire le operazioni da fargli fare in base a quello che si dice
