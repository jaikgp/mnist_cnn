'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
from random import shuffle
import tensorflow as tf


sess = tf.Session()

n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

def shuffle_data(trainx,trainy):
    c = zip(trainx,trainy)
    shuffle(c)
    return zip(*c)

def prepare_x(trainX,trainY):
    trainx = np.array([trainX[idx].reshape((784)).astype(np.float) for idx in xrange(len(trainX))])
    #trainy = np.array([trainY[idx] for idx in xrange(len(trainY))])
    trainy = []
    for idx in xrange(len(trainY)):
        y = np.zeros(10)
        y[trainY[idx]]=1
        trainy.append(y)
    trainy = np.array(trainy)

    c = zip(trainx,trainy)
    shuffle(c)
    return zip(*c)


def model_build(x,weights):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), weights['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), weights['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + weights['outb']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),

    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'outb': tf.Variable(tf.random_normal([n_classes]))
}


def train(trainX, trainY):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32,shape=[None,10])

    pred = model_build(x,weights)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=pred))
    #cross_entropy = tf.reduce_sum(- y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) - (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)), 1)
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.initialize_all_variables()
    
    sess.run(init_op)

    trainx,trainy = prepare_x(trainX,trainY)
    val_trainx = trainx[int(len(trainx)*0.8):]
    val_trainy = trainy[int(len(trainy)*0.8):]
    trainx = trainx[:int(len(trainx)*0.8)]
    trainy = trainy[:int(len(trainy)*0.8)]

    for epoch in xrange(50):
        trainx,trainy = shuffle_data(trainx,trainy)
        for batch in xrange(len(trainx)/100):
            sess.run(train_step,{x: trainx[batch*100:(batch+1)*100], y_: trainy[batch*100:(batch+1)*100]})
        print(sess.run(accuracy,{x: val_trainx[:], y_: val_trainy[:]}))

    parameters = weights.copy()
    saver = tf.train.Saver(parameters)
    save_path = saver.save(sess, "Weights/models_dense/model2.ckpt")

    #sess.close()

# def train(trainX, trainY):
#     x = tf.placeholder(tf.float32, shape=[None, 784])
#     y_ = tf.placeholder(tf.float32,shape=[None,10])

#     W = tf.Variable(tf.zeros([784,10]))
#     b = tf.Variable(tf.zeros([10]))

#     y = tf.matmul(x,W) + b
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
#     train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#     correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#     init_op = tf.initialize_all_variables()
#     sess = tf.Session()
#     sess.run(init_op)

#     trainx,trainy = prepare_x(trainX,trainY)
#     val_trainx = trainx[int(len(trainx)*0.8):]
#     val_trainy = trainy[int(len(trainy)*0.8):]
#     trainx = trainx[:int(len(trainx)*0.8)]
#     trainy = trainy[:int(len(trainy)*0.8)]

#     for epoch in xrange(50):
#         trainx,trainy = shuffle_data(trainx,trainy)
#         for batch in xrange(len(trainx)/100):
#             sess.run(train_step,{x: trainx[batch*100:(batch+1)*100], y_: trainy[batch*100:(batch+1)*100]})
#         print(sess.run(accuracy,{x: val_trainx[:], y_: val_trainy[:]}))

#     sess.close()


def test(testX):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32,shape=[None,10])

    parameters = weights.copy()
    saver = tf.train.Saver(parameters)
    saver.restore(sess,"Weights/models_dense/model2.ckpt")

    #new_weights = {

    #}

    testx = np.array([testX[idx].reshape((784)).astype(np.float32) for idx in xrange(len(testX))])

    #sess = tf.Session()
    pred = model_build(testx,parameters).eval(session=sess)
    labels = np.array([np.argmax(pre) for pre in pred])


    sess.close()

    return labels
