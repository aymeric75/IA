import numpy as np
import keras



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, training_examples, labels, indices_word, word_indices,  maxlen, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.indices_word = indices_word
        self.training_examples = training_examples
        self.shuffle = shuffle
        self.word_indices = word_indices
        self.maxlen = maxlen
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.training_examples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        training_examples_temp = [self.training_examples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(training_examples_temp, indexes)



        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.training_examples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, training_examples_temp, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.maxlen, len(self.indices_word)), dtype=np.bool)
        Y = np.zeros(((self.batch_size), len(self.indices_word)), dtype=np.bool)

        for i, sentence in enumerate(training_examples_temp):
            for t, word in enumerate(sentence):
                X[i, t, self.word_indices[word]] = 1
            Y[i, self.word_indices[self.labels[indexes[i]]]] = 1


        return X, Y