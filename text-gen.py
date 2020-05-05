from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from my_classes import DataGenerator
import numpy as np
import random
import sys
import io




def return_tokenized_text(text):

    special_words=[",",";",".",":"]
    text_list = []

    with open(text, "r", encoding='utf-8') as a_file:
        for i, line in enumerate(a_file):
            
            line_splitted = line.split(" ")

            for j, word in enumerate(line_splitted):
                word = word.strip("\t")
                if([ele for ele in special_words if(ele in word)]):
                    if(word[:-1] != ""):
                        text_list.append(word[:-1])
                    if(word[-1] != ""):
                        text_list.append(word[-1])
                elif("(" in word and ")" in word):
                    text_list += ["(", word[1:-1], ")"]
                elif("(" in word and ")" not in word):
                    text_list += ["(", word[1:]]
                elif(")" in word and "(" not in word):
                    text_list += [ word[:-1], ")"]
                elif("\n" in word and word != "\n"):
                    text_list += [ word[:-1], "\n"]
                else:
                    if(word != ""):
                        text_list.append(word)

        without_empty_strings = []

        without_empty_strings  = [string for string in text_list if string != ""]

        return without_empty_strings 





# tokenized version of the french text
new_text = return_tokenized_text("./poetry-french.txt")


# list of all words present in the text
words = sorted(list(set(new_text)))



# self explanatory...
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))




maxlen = 250 # lenght (<=> number of words) of each training example
step = 30 
X = []       # List of all training examples
Y = []       # List all corresponding labels
for i in range(0, len(new_text) - maxlen, step):
    X.append(new_text[i: i + maxlen])
    Y.append(new_text[i + maxlen])




# Parameters of the generator
params = {
          'batch_size': 12,
          'shuffle': True,
          'word_indices': word_indices,
          'indices_word': indices_word,
          'maxlen': maxlen
          }


training_generator = DataGenerator(X, Y, **params)


model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(words))))
model.add(Dense(len(words), activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)




def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(new_text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = new_text[start_index: start_index + maxlen]
        generated += str(sentence)
        print('----- Generating with seed: "' + str(sentence) + '"')
        #sys.stdout.write(generated)

        for i in range(40):
            x_pred = np.zeros((1, maxlen, len(words)))
            for t, char in enumerate(sentence):
                x_pred[0, t, word_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]

            next_index = sample(preds, diversity)
            next_char = indices_word[next_index]
            sentence = sentence[1:] + [next_char]

            sys.stdout.write(next_char+"  ")
            sys.stdout.flush()
        print()




print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit_generator(generator=training_generator, steps_per_epoch = 6, epochs=22, callbacks=[print_callback])