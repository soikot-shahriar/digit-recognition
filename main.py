import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train  = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x=x_train,y=y_train, epochs=3)

test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
print("\n Test accuracy:", test_acc)
print("\n Test loss:", test_loss)

prediction = model.predict([x_test])
print("DIGIT = ",np.argmax(prediction[1000]))
plt.imshow(x_test[1000], cmap="gray")
plt.show()