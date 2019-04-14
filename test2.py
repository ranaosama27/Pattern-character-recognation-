from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
from subprocess import check_output
import os
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from time import time
from time import sleep
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from tensorflow.keras import datasets, layers, models



DataDir="E:\Level 3\Pattern\English\Fnt"
Categories=["Zero","One","Two","Three"]#,"Four","Five","six","Seven","Eight","Nine","A-Capital","B-Capital","C-Capital","D-Capital","E-Capital","F-Capital","G-Capital","H-Capital","I-Capital",
            #"J-Capital","K-Capital","L-Capital","M-Capital","N-Capital","O-Capital","P-Capital","Q-Capital","R-Capital","S-Capital","T-Capital","U-Capital","V-Capital","W-Capital","X-Capital","Y-Capital","Z-Capital",
            #"a-Small","b-Small","c-Small","d-Small","e-Small","f-Small","g-Small","h-Small","i-Small","j-Small","k-Small","l-Small","m-Small","n-Small","o-Small","p-Small","q-Small","r-Small","s-Small","t-Small",
            #"u-Small","v-Small","w-Small","x-Small","y-Small","z-Small"]

training_data=[]
image_size=28
x=[]
y=[]
def create_trainig_data():
    for category in Categories:
        path=os.path.join(DataDir,category)
        class_num=Categories.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array=cv2.resize(img_array,(image_size,image_size))
            training_data.append([new_array,class_num])
create_trainig_data()
random.shuffle(training_data)
for features,label in training_data:
   x.append(features)
   y.append(label)    
x=np.array(x).reshape(-1,image_size,image_size,1)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, batch_size=16, epochs=3, validation_split=0.2)
#model.summary()