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

    def predictIntention(self, sentence):
        predizione_intento = self.interpreter.parse(sentence)['intent']
        confidenza = predizione_intento['confidence']
        intent = predizione_intento['name']
        toReturn = {'confidenza': confidenza, 'intento': intent}
        return toReturn