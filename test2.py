import cv2
import numpy as np
import os
from random import shuffle 
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


LR=1e-3
model_name='characters'.format(LR,'2conv-basic')
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
shuffle(training_data)
for features,label in training_data:
   x.append(features)
   y.append(label)

convnet = input_data(shape=[None,image_size,image_size,1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

train=training_data[:300]
test=training_data[300:]    
x=np.array(x).reshape(-1,image_size,image_size,1)
y=[i[1]for i in train]

test_x=np.array([i[0]for i in test]).reshape(-1,image_size,image_size,1)
test_y=[i[1] for i in test]

model.fit({'input': x}, {'targets': y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=model_name)


