# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:21:54 2019

@author: win8
"""


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
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#
# DataDir = "E:\Pattern-character-recognition\Dataset"
# Categories = ["Zero","One","Two","Three","Four","Five","six","Seven","Eight","Nine","A-Capital","B-Capital","C-Capital","D-Capital","E-Capital","F-Capital","G-Capital","H-Capital","I-Capital",
#               "J-Capital","K-Capital","L-Capital","M-Capital","N-Capital","O-Capital","P-Capital","Q-Capital","R-Capital","S-Capital","T-Capital","U-Capital","V-Capital","W-Capital","X-Capital","Y-Capital","Z-Capital",
#               "a-Small","b-Small","c-Small","d-Small","e-Small","f-Small","g-Small","h-Small","i-Small","j-Small","k-Small","l-Small","m-Small","n-Small","o-Small","p-Small","q-Small","r-Small","s-Small","t-Small",
#               "u-Small","v-Small","w-Small","x-Small","y-Small","z-Small"]
# training_data = []
# image_size = 50
# x = []
# y = []
#
#
# def create_trainig_data():
#     for category in Categories:
#         path = os.path.join(DataDir, category)
#         class_num = Categories.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (image_size, image_size)).flatten()
#                 training_data.append([new_array, class_num])
#                 # plt.imshow(new_array)
#                 # plt.show()
#                 # break
#             except Exception as e:
#                 pass
#         # break
#
#
# create_trainig_data()
# for features, label in training_data:
#     x.append(features)
#     y.append(label)
# random.shuffle(training_data)
#
#
# pickle_out = open("x.pickle","wb")
# pickle.dump(x,pickle_out)
# pickle_out.close()
#
#
# pickle_out = open("y.pickle","wb")
# pickle.dump(y,pickle_out)
# pickle_out.close()

pickle_in = open("x.pickle","rb")
x = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
# print(x[1])
x =np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)



# x = np.array(x)
# x = x.flatten()
# x = x.tolist()
# print(x.shape)
# x = np.array(x).reshape(-1, image_size, image_size,1)
# x =np.array(x)
# print(x.shape)
# for i in x():
#  print(i)
# print(y)
# print(y.shape)


# x =x.reshape(-1, image_size, image_size,1)
# y = np.array(y).reshape(-1,1)
#
# print(x.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score)
