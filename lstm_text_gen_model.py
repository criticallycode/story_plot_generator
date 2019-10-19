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