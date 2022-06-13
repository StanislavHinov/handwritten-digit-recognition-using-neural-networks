import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load the trained model

model = tf.keras.models.load_model('handwritten.model')

# Evaluate model, test data is neeed, we get the same dataset as the train script.

mnist = tf.keras.datasets.mnist

# Loads the MNIST dataset - https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
# The default Train-Test Split is 80/20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixels, grayscale pixel can have lightness from 0-255, when we normalize it the values range from 0-1,
# this makes it easier for the NN to make calculations

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

loss, accuracy = model.evaluate(x_test, y_test)


print(loss)
print(accuracy)

