import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# add dataset of handwritten numbers which is included in tensorflow, the dataset is called MNIST
# https://www.tensorflow.org/datasets/catalog/mnist

mnist = tf.keras.datasets.mnist

# Loads the MNIST dataset - https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
# The default Train-Test Split is 80/20
# x_train: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. Pixel values range from 0 to 255.
# y_train: uint8 NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.
# x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. Pixel values range from 0 to 255.
# y_test: uint8 NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixels, grayscale pixel can have lightness from 0-255, when we normalize it the values range from 0-1,
# this makes it easier for the NN to make calculations

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Preparation of data is done, we can start working on NN
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

# Create our model, which is sequential, one layer leads to the next one
model = tf.keras.models.Sequential()

# Creating a Flattening layer from matrices to arrays
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer
# It's the most basic layer in neural networks.
# https://www.youtube.com/watch?v=68BZ5f7P94E
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Output layer
# Softmax - gives the probability of each digit to be the right answer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# We compile the module
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train module, epochs how many times is the model gonna see the same data

model.fit(x_train, y_train, epochs=10)

# Save the module

model.save('handwritten.model')
