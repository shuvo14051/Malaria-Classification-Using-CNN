import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import tensorflow as tf
from tensorflow import keras



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread


# # Displaying Uninfected and Infected Cell tissues


import cv2
upic='../input/cell-images-for-detecting-malaria/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_131.png'
apic='../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_164.png'
plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(cv2.imread(upic))
plt.title('Uninfected Cell')
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(cv2.imread(apic))
plt.title('Infected Cell')
plt.xticks([]) , plt.yticks([])

plt.show()


width = 128
height = 128


# # Dividing Dataset into two folders train and test


datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)


# # Preparing train and test Image Generator

trainDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                           target_size=(width,height),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='training')


valDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                           target_size=(width,height),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='validation',
                                           shuffle=False)

# # Preparing the model

from tensorflow import keras
model = keras.models.Sequential([
keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape = [128,128,3], padding='same'),
keras.layers.MaxPooling2D(),
keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
keras.layers.MaxPooling2D(),

keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same'),
keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same'),
keras.layers.BatchNormalization(),
keras.layers.MaxPool2D(),

keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same'),
keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same'),
keras.layers.BatchNormalization(),
keras.layers.MaxPool2D(),
keras.layers.Dropout(0.2),

keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same'),
keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same'),
keras.layers.BatchNormalization(),
keras.layers.MaxPool2D(),
keras.layers.Dropout(0.2),

keras.layers.Flatten(),

keras.layers.Dense(512, activation='relu'),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.7),

keras.layers.Dense(128, activation='relu'),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.5),

keras.layers.Dense(64, activation='relu'),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.3),

keras.layers.Dense(1, activation ='sigmoid')])

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
callbacks = [early_stopping, reduce_lr]

history = model.fit(trainDatagen,
                    epochs =50,
                    validation_data = valDatagen,
                    callbacks=callbacks, 
                    batch_size=128)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

fig, ax = plt.subplots(1, 2, figsize = (30, 10))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

model.evaluate(valDatagen)

# df = pd.DataFrame(history.history)
# df.to_csv("Malaria-CNN.csv")

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
preds = model.predict_generator(valDatagen)
y_pred = tf.where(preds<=0.5,0,1)
y_true = valDatagen.labels
print(classification_report(y_true,y_pred))

import seaborn as sns
cnf = confusion_matrix(y_true,y_pred)
sns.heatmap(cnf, annot=True,fmt='g', cmap="PuRd")
plt.savefig("CNN-Confusion matrix.png")
plt.show()


# # Specificity and sensitivity of CNN

tp = 2542
fp = 213
tn = 2684 
fn = 71
sensitivity = tp/(tp+fn)
print(sensitivity)
specificity = tn/(tn+fp)
print(specificity)


# # VGG16 Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

vgg = VGG16(input_shape=(128,128,3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)


prediction = Dense(1, activation='sigmoid')(x)

modelvgg = Model(inputs=vgg.input, outputs=prediction)
modelvgg.summary()

METRICS = ['accuracy']

modelvgg.compile(optimizer='adam',
              loss=['binary_crossentropy'],
              metrics=METRICS)

history = modelvgg.fit(trainDatagen,
                    epochs =50,
                    validation_data = valDatagen,
                    callbacks=callbacks, 
                    batch_size=128)

fig, ax = plt.subplots(1, 2, figsize = (30, 10))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

modelvgg.evaluate_generator(valDatagen)

df = pd.DataFrame(history.history)

# df.to_csv("Malaria-VGG.csv")

# # Perfromance

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
preds = modelvgg.predict_generator(valDatagen)
y_pred = tf.where(preds<=0.5,0,1)

y_true = valDatagen.labels
print(classification_report(y_true,y_pred))

import seaborn as sns
cnf_vgg = confusion_matrix(y_true,y_pred)
sns.heatmap(cnf_vgg, annot=True,fmt='g',cmap="GnBu")
plt.savefig("VGG-Confusion matrix.png")
plt.show()


# # Specificity and sensitivity of VGG16

tp = 2631
fp = 124
tn = 2500 
fn = 255
sensitivity = tp/(tp+fn)
print(sensitivity)
specificity = tn/(tn+fp)
print(specificity)

import os
os.system('uname -o')

import sys
print(sys.platform)


