import sys
sys.path.insert(0,'/usr/local/lib/python3.5/site-packages/')
import tensorflow as tf

import random
from layer import *
from loadDataset import *
import numpy as np
import math as mt

'''
Need to insert checkpoints and summaries in this project.
'''

total_epochs = 10
current_epoch = 0
batch_size = 2          #24
num_frames = 16
state_size = 256
num_classes = 101
lrate = 0.0001
total_size = 4     # number of videos in one hdf5 file This is required as in the data iterator I am shuffling the values from 1 to total videos
                    # so that training is not done always
                   # in the same order. As I am getting error after 5 so for running I am taking 4 as total size.
#check I have removed expand _dimfor batchsize as well as I have taken the shape of input to be None,227,227,3 and labels as None. Is none allowed.
#if I take None, 16, 227,227,3 I am getting error in convolution as filter has size 3 but input has 5.
#Need to figure out what the data iterator must be. I am not being able to figure out 
#what is 120 in the output of hdf5 as well as how to make batches as we have just 1 video
#allof whose frames has to be given to the lstm

class SimpleDataIterator(object):
    def __init__(self, total_size):
        self.size = total_size
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.lst = [i for i in range(self.size)]
        random.shuffle(self.lst)
        self.cursor = 0

    def next_batch(self, batch_size):
        if self.cursor+batch_size-1 > self.size:
            self.epochs += 1
            self.shuffle()
        lst_new = self.lst[self.cursor:self.cursor+batch_size]
        for index in lst_new:
            x, y, _ = load_dataset(vidsFile="./trainlist01.txt",
                                   baseDataPath="/home/abhishek/tensorflow/Projects/lrcn",
                                   index=index,
                                   fName="trainlist",
                                   frameTotal=16,
                                   size=227,
                                   isTraining=True,
                                   classIndFile='./classInd.txt',
                                   chunk=4,
                                   Type='normal',
                                   k=20)
            try:
                x_batch = np.concatenate((x_batch,x), axis=0)
                y_batch = np.concatenate((y_batch,y), axis=0)
            except:
                x_batch = np.copy(x)
                y_batch = np.copy(y)
        self.cursor += batch_size
        return x_batch, y_batch


#x = tf.expand_dims(x, 0)
#y = tf.expand_dims(y, 0)
#Placeholders

images = tf.placeholder(name="image", shape=[None, 227, 227, 3], dtype=tf.float32)
labels = tf.placeholder(name="labels", shape=[None], dtype=tf.int32)

#Model
def model(input):
    net = conv_layer(input, filter_shape=[7,7,3,96], stride_value=[1,2,2,1], scope_name="conv1")
    net = pool_layer(net, ksize=[1,3,3,1], stride=[1,2,2,1], scope_name="pool1")
    net = tf.nn.local_response_normalization(net, depth_radius=5, bias=None, alpha=0.0001, beta=0.75)
    net = conv_layer(net, filter_shape=[5,5,96,384], stride_value=[1,2,2,1], scope_name="conv2")
    net = pool_layer(net, ksize=[1, 3, 3, 1], stride=[1, 2, 2, 1], scope_name="pool2")
    net = tf.nn.local_response_normalization(net, depth_radius=5, bias=None, alpha=0.0001, beta=0.75)
    net = conv_layer(net, filter_shape=[3,3,384,512], stride_value=[1,1,1,1], scope_name="conv3", padding="SAME")
    net = conv_layer(net, filter_shape=[3,3,512,512], stride_value=[1,1,1,1], scope_name="conv4", padding="SAME")
    net = conv_layer(net, filter_shape=[3,3,512,384], stride_value=[1,1,1,1], scope_name="conv5", padding="SAME")
    net = pool_layer(net, ksize=[1, 3, 3, 1], stride=[1, 2, 2, 1], scope_name="pool5")
    net = fc(net, 4096, scope_name="fc6")
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, keep_prob=0.1)
    net = tf.reshape(net, shape=[batch_size,16,4096])  #16 = Num_frames, 24 = batch_size, 4096 = output_channels
    net = lstm(net, state_size)
    net = tf.nn.dropout(net, keep_prob=0.5)
    #net = fc(input=net, output_channels=101, scope_name="fc8", constant_init=0, stddev_init=0.01)   #these are logits
    net = tf.reshape(net, [-1, state_size])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    logits = tf.matmul(net,W)+b
    return logits

def lstm(input, state_size=256):
    num_batch, num_steps, _ = input.get_shape().as_list()
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=input, initial_state=init_state)
    return rnn_outputs

#Calculate loss
logit = model(input=images)
label_reshaped = tf.reshape(labels, [-1])
total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=label_reshaped))

#Training
train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss=total_loss)

#Reset graph
#if 'sess' in globals() and sess:
#    sess.close()
#tf.reset_default_graph()

#Start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    training_losses = []
    data = SimpleDataIterator(total_size)
    while total_epochs>current_epoch:
        training_loss = 0
        steps = 0
        while steps<mt.ceil(total_size/batch_size):
            steps += 1
            x, y = data.next_batch(batch_size)
            loss_, _ = sess.run([total_loss, train_step], feed_dict={images: x, labels: y})
            training_loss += loss_
            print("Training loss for step {} of epoch {} is {}".format(steps,current_epoch,loss_))
        print("Average training loss for Epoch ", current_epoch, ":", training_loss/steps)
        current_epoch += 1
        training_losses.append(training_loss/steps)
