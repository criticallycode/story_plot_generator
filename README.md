# story_plot_generator
This repo demonstrates multiple forms of text generation used to create movie plot ideas.

This project attempts to create story ideas based off of plot summaries to popular movies and TV shows. There are multiple ways that the sample plots are generated. They are generated with an LSTM neural network and Markov chains.

The effectiveness of these plot generators  will be compared.

The samples of generated text you'll see below were generated based on the Horror plots, but plots for other genres are available for use in the repo.

Before we can generate story plots, we need some plots to base our generations on. We'll be using an API to collect data on movies, and storing the data in an SQL database for later processing and refinement. To begin with, we create a database to store the films.

An SQL database was created to hold the results of the web scraping. It takes in title, genre, plot, rating, and year fields.

```Python
import sqlite3

connection = sqlite3.connect('movies_genres.db')

cursor = connection.cursor()

# only run this when creating table
sql_command = """CREATE TABLE movies (title VARCHAR, genre VARCHAR, plot VARCHAR, rated VARCHAR, year VARCHAR)"""
cursor.execute(sql_command)

connection.commit()
connection.close()
```

For the selected genres - in this case I decided on six: Drama, Horror, Thriller, Mystery, Action, and Sci-fi - I pulled the title data from IMDB, saving them into a CSV.

```Python
import requests
from bs4 import BeautifulSoup
import re
import csv
import numpy as np
import time

# genres = Drama, Horror, Thriller, Mystery, Action, Sci-fi

pages = list(range(1,9))
url_base = 'https://www.imdb.com/search/title?genres=Horror&start='
url_end = '&explore=title_type,genres&ref_=adv_nxt'
print(pages)

nums = np.arange(0, 1001, 50)
print(nums)

# handle loop through by incrementing number in groups of 50

for num in nums:
    time.sleep(3)
    complete_url = url_base + str(num) + url_end
    #print(complete_url)
    print("Getting URL data")
    r = requests.get(complete_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    blocks = (soup.find_all('h3', {'class':'lister-item-header'}))
    #text = links.find_all(class_='frame')
    #print(blocks)

    for title in blocks:

        titles = []

        for y in title.find_all('a'):
            titles.append(y.text)

        print(titles)

        with open('Horror_movies.csv', 'a') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(titles)

```

The CSV file had extra blank lines in-between the titles. This functions below remove blank lines from CSVs and save a new file. There is also a function to do the same with text files, as I discovered I had to do this later on.

```Python
import pandas as pd

csv_in = "Horror_movies.csv"
csv_out = "Horror_movies_correct.csv"

#text_in = "Horror_plots_2.txt"
#text_out = "Horror_plots_2_correct.txt"

def csv_remove_blank(text_file, out_file):
    df = pd.read_csv(text_file, encoding = "ISO-8859-1")
    print (df.isnull().sum())
    modifiedDF = df.dropna()
    modifiedDF.to_csv(out_file, index=False)

def text_remove_blank(text_file, out_file):
    with open(text_file) as infile, open(out_file, 'w') as outfile:
        for line in infile:
            if not line.strip(): continue  # skip the empty line
            outfile.write(line)  # non-empty line. Write it to output

csv_remove_blank(csv_in, csv_out)
#text_remove_blank(text_in, text_out)
```

Plot data is pulled from the Open Movie Database API, which can be accessed with the URL below and your own API key. The code takes in the titles pulled from the CSV full of titles and creates a query URL. It then saves all the URLs in a CSV.

```Python
import csv
import re

url_base = "http://www.omdbapi.com/?apikey=[YOUR API KEY GOES HERE]&t="

with open('Horror_movies_correct.csv', mode='r') as data:
    reader = csv.reader(data)
    titles = list(reader)

print(titles)

translation_table1 = dict.fromkeys(map(ord, "]'["), None)

for title in titles:
   concat_url = url_base + str(title)
   concat_url2 = str(concat_url).translate(translation_table1)
   concat_url3 = concat_url2.replace(" ", "+")
   concat_url3 = re.sub(r'"', '', concat_url3)

   #print(concat_url3)

   urls = []
   urls.append(concat_url3)

   print(urls)

   with open ('Horror_urls.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(urls)
```

After using the function to remove blank lines again, requests are made with the CSV full of URLs and the collected data is stored in the database. I did this for all the CSVs of URLs for different genres that I had.

```Python
import sqlite3
import requests
import csv
import re
from simplejson.errors import JSONDecodeError

# create the database
connection = sqlite3.connect("movies.db")

cursor = connection.cursor()

# get API data

file = open('Horror_urls_2_correct.csv')
csv_file = csv.reader(file)

for row in csv_file:
    try:
        url = re.sub(r'.*http', 'http', row[0])
        response = requests.get(url).json()
        print(response)

        title = response['Title']
        genre = response['Genre']
        year = response['Year']
        rating = response['Rated']
        plot = response['Plot']

        print("titles is:", title)
        print("genres are:", genre)

        # title, genre, plot, rated, year
        cursor.execute("INSERT INTO movies (title, genre, plot, rated, year) VALUES (?, ?, ?, ?, ?)", (title, genre, plot, rating, year))

        connection.commit()
        print("Added to database")
    except (KeyError, JSONDecodeError):
        continue


connection.close()
```

The data is then extracted from the SQL database and saved into a CSV, as well as a text file. Either format can be read to generate text with, but I found the text file easier to work with. That said, the CSV of plots did provide an easy way to check the unique numbers of entries.

```Python
import sqlite3
import csv

connection = sqlite3.connect("movies.db")

cursor = connection.cursor()

cursor.execute("SELECT DISTINCT plot FROM movies WHERE genre LIKE '%Horror%'")
with open('Horror_plots.csv', 'w') as p:
        plot_csv_writer = csv.writer(p)
        for row in cursor.fetchall():
            plot_csv_writer.writerow(row)

cursor.execute("SELECT DISTINCT plot FROM movies WHERE genre LIKE '%Horror%'")
with open('Horror_plots.txt', 'w') as pp:
        plot_text_writer = csv.writer(pp)
        for row in cursor.fetchall():
            plot_text_writer.writerow(row)

```

After any blank lines were removed from the text document it was ready to be used in text generation, after some preprocessing of course.

While I have several data files covering different genres available for you to peruse, I will demonstrate the techniques on the list of horror plots.

I experimented with two different methods of generating text based on deep learning. One was a Long Short-Term Memory model created in Keras, while the other is a pre-made model - the TextGenRNN model found [HERE.](https://github.com/minimaxir/textgenrnn)

Here is the code for the custom LSTM model I experimented with:

```Python
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import warnings
warnings.filterwarnings("ignore")

text = open('horror_plots.txt', 'r').read().lower()
print('text length', len(text))

print(text[:300])

chars = sorted(list(set(text)))
print('total chars: ', len(chars))

char_to_num = dict((c, i) for i, c in enumerate(chars))
num_to_char = dict((i, c) for i, c in enumerate(chars))

max_length = 50
step = 3
sentences = []
next = []

for i in range(0, len(text) - max_length, step):
    sentences.append(text[i: i + max_length])
    next.append(text[i + max_length])

print('Number of sequences:', len(sentences))

x = np.zeros((len(sentences), max_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for idx, sentence in enumerate(sentences):
    for i, char in enumerate(sentence):
        x[idx, i, char_to_num[char]] = 1
    y[idx, char_to_num[next[idx]]] = 1

optim = Adam(lr=0.01)

def create_model(max_len, charas, optim):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_len, len(charas)), return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(len(charas)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model

model = create_model(max_length, chars, optim)

def sample(preds, temperature=1.0):
    # pull an index from probability array
    preds = np.asarray(preds).astype('float64')
    # get the log probability
    preds = np.log(preds) / temperature
    # expand the array
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # create a multinomial distribution based on the preds and sample from it
    probs = np.random.multinomial(1, preds, 1)
    # get the max (highest probability) prediction
    return np.argmax(probs)

def end_epoch(epoch, logs):
    # function runs every 5 epochs
    # prints a string of text text_created with the current model parameters

    if epoch % 5 == 0:

        print()
        print('Generation results after epoch {}'.format(epoch))
        print()
        # Set an initial index to start our generation on by randomly selecting an integer
        initial_idx = random.randint(0, len(text) - max_length - 1)
        # Generate the characters for our list of different divesrsities
        for diversity in [0.5, 0.75, 1.0, 1.25]:
            print('Current diversity: {}'.format(diversity))
            print()
            # Reference the starting index and get the sentence that follows
            # This sentence is what will be used to generate the text
            text_created = ''
            sentence = text[initial_idx: initial_idx + max_length]
            text_created += sentence
            print("Seed to generate from:".format(sentence))
            sys.stdout.write(text_created)

            for i in range(500):

                # Construct an array of zeros to fit the predictions into
                feature_pred = np.zeros((1, max_length, len(chars)))

                # Fill in the feature array with the numbers that represent characters
                for n, char in enumerate(sentence):
                    feature_pred[0, n, char_to_num[char]] = 1.

                # Use the model to predict based off of the currents features
                preds = model.predict(feature_pred, verbose=0)[0]
                # get the most likely prediction from the predictions list
                next_idx = sample(preds, diversity)
                # Convert the index to an actual character
                next_char = num_to_char[next_idx]

                # Add the character to the string to be text_created
                text_created += next_char
                # Move on to the next character in the sentence
                sentence = sentence[1:] + next_char

                # Start compiling the list of probable next characters based on sample predictions
                sys.stdout.write(next_char)
                # Write the characters to the terminal
                sys.stdout.flush()

            print()

    else:
        pass

print_callback = LambdaCallback(on_epoch_end=end_epoch)

filepath = "weights.hdf5"

callbacks = [print_callback,
            ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, verbose=1, mode='min', min_lr=0.00001),
            EarlyStopping(monitor= 'loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

records = model.fit(x, y, batch_size=128, epochs=100, callbacks=callbacks)

t_loss = records.history['loss']
t_acc = records.history['acc']

# gets the lengt of how long the model was trained for
train_length = range(1, len(t_loss) + 1)

def evaluation(model, train_length, training_loss, training_acc):

    # plot the loss across the number of epochs
    plt.figure()
    plt.plot(train_length, training_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_length, training_acc, 'r', label='Training acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Accuracy Over Epochs')
    plt.show()

    # compare against the test training set
    # get the score/accuracy for the current model
    scores = model.evaluate(x, y, batch_size=128)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

evaluation(model, train_length, t_loss, t_acc)

def text_gen():

    model.load_weights("weights.hdf5")

    initial_idx = random.randint(0, len(text) - max_length - 1)
    # Generate the characters for our list of different divesrsities
    for diversity in [0.5, 0.75, 1.0, 1.25]:
        print('Current diversity: {}'.format(diversity))
        print()
        # Reference the starting index and get the sentence that follows
        # This sentence is what will be used to generate the text
        text_created = ''
        sentence = text[initial_idx: initial_idx + max_length]
        text_created += sentence
        print("Seed to generate from:".format(sentence))
        sys.stdout.write(text_created)

        for i in range(500):

            # Construct an array of zeros to fit the predictions into
            feature_pred = np.zeros((1, max_length, len(chars)))

            # Fill in the feature array with the numbers that represent characters
            for n, char in enumerate(sentence):
                feature_pred[0, n, char_to_num[char]] = 1.

            # Use the model to predict based off of the currents features
            preds = model.predict(feature_pred, verbose=0)[0]
            # get the most likely prediction from the predictions list
            next_idx = sample(preds, diversity)
            # Convert the index to an actual character
            next_char = num_to_char[next_idx]

            # Add the character to the string to be text_created
            text_created += next_char
            # Move on to the next character in the sentence
            sentence = sentence[1:] + next_char

            # Start compiling the list of probable next characters based on sample predictions
            sys.stdout.write(next_char)
            # Write the characters to the terminal
            sys.stdout.flush()

text_gen()
```

After 100 epochs of training here was a sample output:

```
"A young girl who has sent to fight part of the psychiatram for a story of the paration to his town who seems to find them to stop to stay authoraties that they design the planning to his friend on his farmer that shows up to realize their 'seemed (1990) and the slaurching frank when they arrive..."
```

This wasn't giving me the kinds of results I wanted so I experimented with another method.

Here's the text code for the creation of the model based on [TextGenRNN.](https://github.com/minimaxir/textgenrnn)

```Python
from textgenrnn import textgenrnn

input_file = "horror_plots.txt"
epochs = 50
weights_file = "textgenrnn_weights.hdf5"

def train_generator(input_file, epochs=0):
    textgen_model = textgenrnn()
    textgen_model.train_from_file(input_file, num_epochs=epochs)
    textgen_model.save("textgenrnn_weights.hdf5")

#train_generator(input_file, epochs)

def gen_text(weights_file):
    textgen_model = textgenrnn()
    textgen_model.load(weights_file)
    textgen_model.generate()

gen_text(weights_file=weights_file)
```

Here was a sample of the output after 160 epochs of training:

```
"A group of lover Sister contests investigating the team of the streets of New York to the Amidated Archia's Owhilog called Sinis Ight Batched finds a building its passes within to activill"

"A mother of two who inherits a boarding slacker after the vampires looking for a dangerous life."

""A teenage boy and his friends face off against a mysterious grave robber known only as the Things of Heisher, where the experiment by a new world, all the while staged into a mysterious go fullinacent.""

"A seemingly innocent man is abducted by a notorious zombie here."

"A young woman finds herself on the receiving end of a terrifying cabin in the woods, hoping to regress venture and rule themselfices to kill his family home funder.""

""A group of death tower lies begin to sleeper the backwoods of an ancient event land in their reignspect-world is that the charged supernatural organt."" 
```

The text generated by this method was a little better, but still didn't make a great deal of sense. 

For the generation with Markov chains, I explored two different methods of generating text. The first method used a custom Markov model based on the work of:

The second method of text generation with Markov models used the Markovify library, available here. I still found the  initial text generated by Markovify somewhat lacking, but the Markovify model can be modified with the use of [en_core_web_sm](https://spacy.io/models/en) to enable the tracking of parts of speech and generate higher quality text.

Here is the code for the custom Markov model.

```Python
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
```

Here is where Markovify is used. Observe that the vanilla Markovify is used, and then the Markovify model is improved with POS-tagging from [en_core_web_sm.](https://spacy.io/models/en)

```Python
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
```

An example of the text generated by the model:

```
Basic Markov model generated:
the world at stake, good and evil strain of vampirism ravages the city until she makes contact with her sister..." "A group of people are put through a series of
_______
Vanilla Markovify:
---
Sentence Gen:
None
A girl is trapped inside her family's lakeside retreat and becomes unable to contact the dead in order to avoid paying his debt to the Devil.
A group of people are trapped in an underwater cave network.
A secret agent exacts revenge on a serial killer and they try to find their way home.
A homicide detective discovers he is a descendant from a line of werewolves.
A small group of military officers and scientists dwell in an underground bunker as they seek to find a dark magic item of ultimate power before a diabolical tyrant can.
A scientist sends a man with a diagnosed 23 distinct personalities.
Two brothers find themselves lost in a mysterious land and try to find the killer before the killer finds them.
A crew of space explorers embark on a mission to find a dark magic item of ultimate power before a diabolical tyrant can.
None
 
Short sentence gen:
A picture-perfect family is shattered when the work of a serial killer through a series of grisly murders.
A nerdy florist finds his chance for success and romance with the help of officer Davis Tubbs to help stop the monster's eating spree.
Five women are stalked by an unknown supernatural force after a sexual encounter.
A crew of space explorers embark on a journey of survival with a special young girl named Melanie.
A woman who lives in a haunted house.
Japan is plunged into chaos upon the appearance of a giant man-eating plant who demands to be fed.
An illustrious British boarding school becomes a bloody battleground when a mysterious person contacts her via text message.
The Critters return to Earth searching for a young boy trying to spy on his babysitter.
Three teenage thieves infiltrate a mansion dinner party secretly hosted by a group of strangers while the demons continue their attack.
Paranormal investigators Ed and Lorraine Warren travel to North London to help a family terrorized by a dark presence in their farmhouse.
________
Modified Markov model:
---
Markov full sentence gen:
 
A group of scientists try to track down the vampires who kidnapped his niece .
None
Chief Brody 's widow believes that her lakeside Vermont home is haunted .
A trio of female reporters find themselves staying overnight in a house plagued by a supernatural spirit named Mama has latched itself to their family .
None
A young governess for two children becomes convinced that the home is haunted by a host of demonic ghosts .
None
The Critters return to Earth searching for a young woman how to become a complete weapon .
None
Two brothers find themselves lost in a mysterious land and try to find the killer before the killer finds them .
________
Markov short sentence gen:
 
An institutionalized young woman becomes terrorized by a ghost - or that she is losing her mind .
A small group of military officers and scientists dwell in an underground bunker as they seek to find a mysterious alien ship .
Two brothers find themselves lost in the territory of the deadliest shark species in the claustrophobic labyrinth of submerged caves .
A secret agent exacts revenge on a serial killer whose work dates back to the 1960s .
Later she is approached by a group of preppy college students .
A brash and arrogant podcaster gets more than he bargained for when a job at a haunted county estate gets out of hand .
A young boy and a bunch of misfit friends embark on a mission to find a dark magic item of ultimate power before a diabolical tyrant can .
A husband and wife who recently lost their baby adopt a 9 year - old daughter is a heartless killer .
A fun weekend turns into madness and horror for a bunch of misfit friends embark on a mission to find a cure in a world overrun by zombies .
Three teenage thieves infiltrate a mansion dinner party secretly hosted by a group of teenagers being targeted by another shark in search of revenge .
```


