from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import cv2
import os
from random import shuffle 
import pickle


tf.logging.set_verbosity(tf.logging.INFO)

DataDir="English/Fnt"
Categories=["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine",
            "A-Capital","B-Capital","C-Capital","D-Capital","E-Capital","F-Capital","G-Capital","H-Capital","I-Capital",
            "J-Capital","K-Capital","L-Capital","M-Capital","N-Capital","O-Capital","P-Capital","Q-Capital","R-Capital",
            "S-Capital","T-Capital","U-Capital","V-Capital","W-Capital","X-Capital","Y-Capital","Z-Capital",
            "a-Small","b-Small","c-Small","d-Small","e-Small","f-Small","g-Small","h-Small","i-Small","j-Small","k-Small",
            "l-Small","m-Small","n-Small","o-Small","p-Small","q-Small","r-Small","s-Small","t-Small",
            "u-Small","v-Small","w-Small","x-Small","y-Small","z-Small"]

training_data=[]
image_size=50
x_train=[]
y_train=[]

x_test=[]
y_test=[]

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


#for features,label in training_data:
#  x.append(features)
#  labels.append(label)


train=training_data[:52000]
test=training_data[52000:] 
  
for features,label in train:
  x_train.append(features)
  y_train.append(label)


for features,label in test:
  x_test.append(features)
  y_test.append(label)
  
  
  
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
  
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": x_train},
      y=y_train,
      batch_size=100,
      num_epochs=2,
      shuffle=True)

eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": x_test}, y=y_test, num_epochs=1, shuffle=False)
eval_results=training_data.evaluate(eval_input_fn)
print(eval_results)