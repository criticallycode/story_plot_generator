import markovify
import en_core_web_sm
import markovgen2_works as Markov

markov_data = open("Horror_plots.txt")
generator = Markov.Markov(2)
generator.train(markov_data)
print("Basic Markov model generated:")
generator.generate(30)
print("_______")

input_text = open('Horror_plots.txt').read()
nlp = en_core_web_sm.load()

# regular markovify

# Build the model.
text_model = markovify.Text(input_text, state_size=2)

# Print five randomly-generated sentences
print("Vanilla Markovify:")
print("---")
print("Sentence Gen:")
for i in range(10):
    print(text_model.make_sentence())

print(" ")
print("Short sentence gen:")
# Print three randomly-generated sentences of no more than 140 characters
for i in range(10):
    print(text_model.make_short_sentence(140))

print("________")

#sentence_gen()

# overwrite default markovify model

class POSText(markovify.Text):
    def word_split(self, sentence):
        return ["::".join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence

text_model2 = POSText(input_text, state_size=3)

print("Modified Markov model:")
print("---")
print("Markov full sentence gen:")
print(" ")
for i in range(10):
    print(text_model2.make_sentence())
print("________")

print("Markov short sentence gen:")
print(" ")
for i in range(10):
    print(text_model2.make_short_sentence(150))
print("________")