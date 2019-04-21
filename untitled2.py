        # -*- coding: utf-8 -*-
    # Import libraries
        #matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model_name = 'characters'.format(LR, '2conv-basic')
DataDir = "English/Fnt"
Categories = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
"A-Capital", "B-Capital", "C-Capital", "D-Capital", "E-Capital", "F-Capital", "G-Capital",
"H-Capital", "I-Capital",
"J-Capital", "K-Capital", "L-Capital", "M-Capital", "N-Capital", "O-Capital", "P-Capital",
"Q-Capital", "R-Capital",
"S-Capital", "T-Capital", "U-Capital", "V-Capital", "W-Capital", "X-Capital", "Y-Capital",
"Z-Capital","a-Small", "b-Small", "c-Small", "d-Small", "e-Small", "f-Small", "g-Small", "h-Small", "i-Small","j-Small", "k-Small",
"l-Small", "m-Small", "n-Small", "o-Small", "p-Small", "q-Small", "r-Small", "s-Small", "t-Small",
"u-Small", "v-Small", "w-Small", "x-Small", "y-Small", "z-Small"]

training_data = []
image_size = 28
x = []
y = []
x_train=[]
x_test=[]
y_train=[]
y_test=[]

def create_trainig_data():
    for category in Categories:
        path = os.path.join(DataDir, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (image_size, image_size))
            training_data.append([new_array, class_num])


create_trainig_data()
shuffle(training_data)

train=training_data[:52000]
test=training_data[52000:]

for features, label in train:
    x_train.append(features)
    y_train.append(label)

for features, label in test:
    x_test.append(features)
    y_test.append(label)

        # Define parameters for the model
learning_rate=0.01
batch_size=128
n_epochs=3

        # MNIST data input (img shape: 28*28)
n_input = 28

        # MNIST total classes (0-9 digits)
n_classes = 62

#train_X = x_train.reshape(-1, 28, 28, 1)
#test_X = x_test.reshape(-1,28,28,1)
#train_y = y_train.labels
#test_y = y_test.labels

x= tf.placeholder("float",[None,28,28,1])
y = tf.placeholder("float",[None,n_classes])


def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding="SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")

weights = {
        'wc1':tf.get_variable('w0',shape=(3,3,1,32),initializer=tf.contrib.layers.xavier_initializer()),
        'wc2':tf.get_variable('w1',shape=(3,3,32,64),initializer=tf.contrib.layers.xavier_initializer()),
        'wc3':tf.get_variable('w2',shape=(3,3,64,128),initializer=tf.contrib.layers.xavier_initializer()),
        'wd1':tf.get_variable('w3',shape=(4*4*128,128),initializer=tf.contrib.layers.xavier_initializer()),
        'out':tf.get_variable('w6',shape=(128,n_classes),initializer=tf.contrib.layers.xavier_initializer()),
        }
biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
        }

def conv_net(x,weights,biases):
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,k=2)
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2,k=2)
    conv3 = conv2d(conv2,weights['wc3'],biases['bc3'])
    conv3 = maxpool2d(conv3,k=2)
            #Fully Connected Layer
    fc1 = tf.reshape(conv3,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

pred = conv_net(x,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

        #Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        #calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    n_batches = int(training_data.num_examples/batch_size)
    for i in range(n_epochs):
        for batch in range(n_batches):
            X_batch,Y_batch = training_data.next_batch(batch_size)

        #            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
        #            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
        #            # Run optimization op (backprop).
                        # Calculate batch loss and accuracy
opt = sess.run(optimizer, feed_dict={x: X_batch,y: Y_batch})
loss, acc = sess.run([cost, accuracy], feed_dict={x: X_batch, y: Y_batch})
print("Iter " + str(i) + ", Loss= " + \
      "{:.6f}".format(loss) + ", Training Accuracy= " +\
      "{:.5f}".format(acc))
print("Optimization Finished!")

                # Calculate accuracy for all 10000 mnist test images
test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
train_loss.append(loss)
test_loss.append(valid_loss)
train_accuracy.append(acc)
test_accuracy.append(test_acc)
print("Testing Accuracy:","{:.5f}".format(test_acc))
summary_writer.close()