from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer
from rasa.nlu import config

class NaturalLanguageUnderstandingModule:
    def __init__(self):
        training_data = load_data("./data/nlu.md")
        trainer = Trainer(config.load("config.yml"))
        self.interpreter = trainer.train(training_data)
        model_directory = trainer.persist("./models/nlu", fixed_model_name="current")
        self.music_verbs = ['Riproduci', 'Suona', 'Fai partire', 'Avvia']

    def predictIntention(self, sentence):
        predizione_intento = self.interpreter.parse(sentence)['intent']
        confidenza = predizione_intento['confidence']
        intent = predizione_intento['name']
        toReturn = {'confidenza': confidenza, 'intento': intent}
        return toReturn

    def format_query_uri(self, sentence):
        indice = -1
        verbo_da_utilizzare = None
        for verbo in self.music_verbs:
            try:
                indice = sentence.lower().index(verbo.lower())
                verbo_da_utilizzare = verbo.lower()
                break
            except ValueError:
                continue
        return ''.join(sentence.lower().split(verbo_da_utilizzare)[1:]).strip().replace(" ","+") if verbo_da_utilizzare is not None else sentence.replace(" ", "+")