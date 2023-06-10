import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_images = np.loadtxt("drive/MyDrive/CIH Intership/emnist-balanced-train.csv", delimiter=",")

print(train_images.shape)


train_images = train_images[np.in1d(train_images[:, 0], (18, 24, 26, 29, 44), invert=True)]
print(train_images.shape)

train_labels = train_images[:, 0]
train_images = train_images[:, 1:]

np.unique(train_labels)

test_images = np.loadtxt("drive/MyDrive/CIH Intership/emnist-balanced-test.csv", delimiter=",")
test_images = test_images[np.in1d(test_images[:, 0], (18, 24, 26, 29, 44), invert=True)]
test_images.shape

test_labels = test_images[:, 0]
test_images = test_images[:, 1:]
print(test_images.shape)
print(test_labels.shape)

by_merge_map = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 
                12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 
                23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 
                34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 
                45:'r', 46:'t'}

"""Transpose image for correct representation"""

train_images = np.transpose(np.reshape(train_images, (train_images.shape[0],28, 28)), axes=[0,2,1])
test_images = np.transpose(np.reshape(test_images, (test_images.shape[0],28, 28)), axes=[0,2,1])

"""Normalize data"""

train_images = keras.utils.normalize(train_images, axis = 1)
test_images = keras.utils.normalize(test_images, axis = 1)

"""Add dimension for single color channel"""

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

print(train_images.shape)

"""Discover examples of images"""

for i in range(0, 3):
    plt.imshow(train_images[i, :, :, 0],cmap = plt.cm.binary)
    plt.xlabel(by_merge_map[train_labels[i]])
    plt.show()

rotation_range_val = 10
width_shift_val = 0.10
height_shift_val = 0.10

train_datagen = ImageDataGenerator(rotation_range = rotation_range_val, width_shift_range = width_shift_val,
                                    height_shift_range = height_shift_val)

train_datagen.fit(train_images.reshape(train_images.shape[0], 28, 28, 1))

val_datagen = ImageDataGenerator()
val_datagen.fit(test_images.reshape(test_images.shape[0], 28, 28, 1))

"""Model defining"""

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(47, activation = 'softmax'))

model.summary()

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

train_images.shape

history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=512),
         validation_data = (test_images, test_labels), validation_batch_size = 32, epochs= 20)

scores = model.evaluate(test_images,test_labels)

print(f"Accuracy: {scores[1]*100}%")

"""Training and validation accuracy and loss plots"""

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

model.save("VIN_model.h5")

