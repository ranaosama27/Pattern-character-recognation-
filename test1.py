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

DataDir="E:\Level 3\Pattern\English\Fnt"
Categories=["Zero","One","Two","Three","Four","Five","six","Seven","Eight","Nine","A-Capital","B-Capital","C-Capital","D-Capital","E-Capital","F-Capital","G-Capital","H-Capital","I-Capital",
            "J-Capital","K-Capital","L-Capital","M-Capital","N-Capital","O-Capital","P-Capital","Q-Capital","R-Capital","S-Capital","T-Capital","U-Capital","V-Capital","W-Capital","X-Capital","Y-Capital","Z-Capital",
            "a-Small","b-Small","c-Small","d-Small","e-Small","f-Small","g-Small","h-Small","i-Small","j-Small","k-Small","l-Small","m-Small","n-Small","o-Small","p-Small","q-Small","r-Small","s-Small","t-Small",
            "u-Small","v-Small","w-Small","x-Small","y-Small","z-Small"]
training_data=[]
image_size=50
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
#random.shuffle(training_data)
#for features,label in training_data:
 #  x.append(features)
#  y.append(label)



for sample in training_data[:10]:
    print(sample[1])
    
    
x=np.array(x).reshape(-1,image_size,image_size,1)