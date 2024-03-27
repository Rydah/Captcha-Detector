import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import tensorflow as tf
from helpers import resize_to_fit

LETTERS_FOLDER = "ExtractedLetter"
MODEL_FILENAME = "captcha_mode.hdf5"
MODEL_LABEL_FILENAME = "model_labels.dat"

data = []
labels = []

for image_file in paths.list_images(LETTERS_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = resize_to_fit(image, 20, 20)

    image = np.expand_dims(image, axis=2)

    label = image_file.split(os.path.sep)[-2]
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.4, random_state=0)

# one-hot encodings
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

with open(MODEL_LABEL_FILENAME, "wb") as f:
    pickle.dump(lb, f)

model = Sequential()

model.add(tf.keras.layers.Conv2D(20, (5,5), padding="same", input_shape=(20,20,1), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(tf.keras.layers.Conv2D(20, (5,5), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation="relu"))

# output of the 36 numbers and letters possible
model.add(tf.keras.layers.Dense(36, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
print('Training Loss:', history.history['loss'])
print('Training Accuracy:', history.history['accuracy'])
print('Validation Loss:', history.history['val_loss'])
print('Validation Accuracy:', history.history['val_accuracy'])

model.save(MODEL_FILENAME)