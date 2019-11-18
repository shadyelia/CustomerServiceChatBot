import chatBot
import numpy
import random
from flask import request

from flask import Flask
app = Flask(__name__)


class Chat:
    def __init__(self):
        self.model, self.words, self.labels, self.data = chatBot.trainOrLoadModel()

    @app.route('/', methods=['Post'])
    def chat(self):
        input = request.data
        results = self.model.predict([chatBot.bag_of_words(input, self.words)])
        results_index = numpy.argmax(results)
        tag = self.labels[results_index]

        for tg in self.data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return (random.choice(responses))


if __name__ == "__main__":
    app.run()
