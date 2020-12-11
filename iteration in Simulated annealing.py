
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([keras.layers.Dense(units=100, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
ys = np.array([0.0, 1.0, 7.0, 53, 558, 8775, 210663], dtype=float)
model.fit(xs, ys, epochs=1500)
print(int(model.predict([8.0])[0].mean()))
