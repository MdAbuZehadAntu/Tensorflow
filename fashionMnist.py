import tensorflow as tf
import numpy as np
from sklearn import metrics

np.set_printoptions(linewidth=200)

mnis = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnis.load_data()
import matplotlib.pyplot as plt

plt.imshow(training_images[42])
print(training_labels[42])
print(training_images[42])
plt.show()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                             tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
