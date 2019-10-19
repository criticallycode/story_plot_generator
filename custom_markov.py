import random
from nltk.corpus import stopwords
import unidecode
import re

class Markov(object):

    def __init__(self, order):

        # order refers to how far back the process will look or remember

        self.order = order

        # controls the actual size of the word groups to be analyzed
        self.group_size = self.order + 1

        # the training text

        self.text = None

        #graph dictionary will hold the actual information
        self.graph = {}

        return

    def train(self, filename):
        self.text = filename.read().split()

        # this appends the beginning of the text to the end of the text
        # so that it always has something to generate
        self.text = self.text + self.text[:self.order]

        # iterate one by one over text, for the entire range of the text starting
        # from word 0 to the last possible groups of word
        for i in range(0, len(self.text) - self.group_size):

            # key is the few words that came before the value
            key = tuple(self.text[i:i + self.order])
            # value is the word that is coming up now, final word in the sequence
            # order 2 markov chain will have value be word 3
            value = self.text[i + self.order]

            # if the word has already been seen, just append the value to the end of the dict
            if key in self.graph:
                self.graph[key].append(value)
            # if word hasn't been seen before, just add it to value of
            # all words we've seen come after specific word pair
            # save the data
            else:
                self.graph[key] = [value]

    def generate(self, length):

        # index defines where the text generation begins at, picks a randomn start word
        index = random.randint(0, len(self.text) - self.order)

        # result comes after the randomly chosen word
        result = self.text[index: index + self.order]

        for i in range(length):

            # current state is the last few words of the current result
            state = tuple(result[len(result) - self.order:])
            # next word is randomly chosen from possible values in the graph
            next_word = random.choice(self.graph[state])
            # append the value to the result
            result.append(next_word)

        print(" ".join(result[self.order:]))