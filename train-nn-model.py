import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# add dataset of handwritten numbers which is included in tensorflow, the dataset is called MNIST - https://www.tensorflow.org/datasets/catalog/mnist

mnist = tf.keras.datasets.mnist

# Loads the MNIST dataset - https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
# The default Train-Test Split is 80/20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixels, grayscale pixel can have lightness from 0-255, when we normalize it the values range from 0-1,
# this makes it easier for the NN to make calculations

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Preparation of data is done, we can start working on NN
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

model = tf.keras.models.Sequential()

# TODO - look what is flatten layer in NN
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# TODO - Dense layer?
# TODO - relu?
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Output layer
# TODO - Softmax?
# Softmax - gives the probability of each digit to be the right answer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# We compile the module
# TODO - adam & sparse_categorical_crossentropy??
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train module, epochs how many times is the model gonna see the same data

model.fit(x_train, y_train, epochs=10)

# Save the module

model.save('handwritten.model')
