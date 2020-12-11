import tensorflow as tf
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.6):
            print("\nReached 60% accuracy so cancelling traing!")
            self.model.stop_training = True


mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255, X_test / 255

callbacks = myCallback()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, callbacks=[callbacks])
model.evaluate(X_test, y_test)
