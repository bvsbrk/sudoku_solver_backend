import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
responses = to_categorical(responses, 10)
samples = samples.reshape((354, 40, 40, 1))

samples = samples / 255

model = Sequential()
model.add(Conv2D(50, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(40, 40, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

num_classes2 = 128
model.add(Dense(num_classes2, activation='relu'))

model.add(Dense(10, activation='relu'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(samples, responses, epochs=10)

model.save("printed_digit.h5")
