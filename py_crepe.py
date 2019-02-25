from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.initializers import RandomNormal
from tensorflow import one_hot, argmax, float32


class Crepe:
    def _one_hot(self, x):
        return one_hot(x, self.vocab_size, on_value=1.0, off_value=0.0, axis=-1, dtype=float32)


    def _one_hot_outshape(self, in_shape):
        return in_shape[0], in_shape[1], self.vocab_size


    def _hot_one(self, x):
        return argmax(x, axis=1)


    def __init__(self, vocab_size, maxlen):
        self.vocab_size = vocab_size
        self.maxlen = maxlen


    def get_classifier(self, filter_kernels, dense_outputs, nb_filter, cat_output, dropout):
        # Define what the input shape looks like
        inputs = Input(shape=(self.maxlen,), dtype='int64')

        # Holding one-hot encodings in memory is very inefficient.
        # The output_shape of embedded layer will be: batch x maxlen x vocab_size

        embedded = Lambda(self._one_hot, output_shape=self._one_hot_outshape)(inputs)

        initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

        convolutional_0 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0], kernel_initializer=initializer,
                             padding='valid', activation='relu',
                             input_shape=(self.maxlen, self.vocab_size))(embedded)
        pooling_0 = MaxPooling1D(pool_size=3)(convolutional_0)

        convolutional_1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[1], kernel_initializer=initializer,
                              padding='valid', activation='relu')(pooling_0)
        pooling_1 = MaxPooling1D(pool_size=3)(convolutional_1)

        convolutional_2 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[2], kernel_initializer=initializer,
                              padding='valid', activation='relu')(pooling_1)

        convolutional_3 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[3], kernel_initializer=initializer,
                              padding='valid', activation='relu')(convolutional_2)

        convolutional_4 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[4], kernel_initializer=initializer,
                              padding='valid', activation='relu')(convolutional_3)

        convolutional_5 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[5], kernel_initializer=initializer,
                              padding='valid', activation='relu')(convolutional_4)
        pooling_5 = MaxPooling1D(pool_size=3)(convolutional_5)

        fully_connected_5 = Flatten()(pooling_5)

        # Two dense layers
        dense_6 = Dropout(dropout)(Dense(dense_outputs, activation='relu')(fully_connected_5))
        dense_7 = Dropout(dropout)(Dense(dense_outputs, activation='relu')(dense_6))

        # Output dense layer with softmax activation
        outputs = Dense(cat_output, activation='softmax', name='output')(dense_7)

        model = Model(inputs=inputs, outputs=outputs)

        sgd = SGD(lr=0.01, momentum=0.9)
        #adam = Adam(lr=0.001)  # Feel free to use SGD above. I found Adam with lr=0.001 is faster than SGD with lr=0.01
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return model
