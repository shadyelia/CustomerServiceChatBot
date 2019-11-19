import chatBot
import numpy
import random


class ChatServers:
    def __init__(self):
        self.model, self.words, self.labels, self.data = chatBot.trainOrLoadModel()

    def Chat(self, input):
        results = self.model.predict([chatBot.bag_of_words(input, self.words)])
        results_index = numpy.argmax(results)
        tag = self.labels[results_index]

        for tg in self.data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return (random.choice(responses))
