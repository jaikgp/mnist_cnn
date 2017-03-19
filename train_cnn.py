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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def shuffle_data(trainx,trainy):
    c = zip(trainx,trainy)
    shuffle(c)
    return zip(*c)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

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

def model_build(x,weights):
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, weights['W_conv1']) + weights['b_conv1'])
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['W_conv2']) + weights['b_conv2'])
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['W_fc1']) + weights['b_fc1'])

    y_conv = tf.matmul(h_fc1, weights['W_fc2']) + weights['b_fc2']

    return y_conv

weights = {
    'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'b_conv1': tf.Variable(tf.random_normal([32])),

    'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'b_conv2': tf.Variable(tf.random_normal([64])),

    'W_fc1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'b_fc1': tf.Variable(tf.random_normal([1024])),

    'W_fc2': tf.Variable(tf.random_normal([1024,10])),
    'b_fc2': tf.Variable(tf.random_normal([10]))

}



def train(trainX,trainY):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32,shape=[None,10])
    x_image = tf.reshape(x, [-1,28,28,1])
    

    pred = model_build(x,weights)

    # W_conv1 = weight_variable([5, 5, 1, 32])
    # b_conv1 = bias_variable([32])

    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)

    # W_conv2 = weight_variable([5, 5, 32, 64])
    # b_conv2 = bias_variable([64])

    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    # W_fc1 = weight_variable([7 * 7 * 64, 1024])
    # b_fc1 = bias_variable([1024])

    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #dropout
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])

    # y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.initialize_all_variables()
    
    sess.run(init_op)

    trainx,trainy = prepare_x(trainX,trainY)
    val_trainx = trainx[int(len(trainx)*0.8):]
    val_trainy = trainy[int(len(trainy)*0.8):]
    trainx = trainx[:int(len(trainx)*0.8)]
    trainy = trainy[:int(len(trainy)*0.8)]

    for epoch in xrange(15):
        trainx,trainy = shuffle_data(trainx,trainy)
        for batch in xrange(len(trainx)/100):
            sess.run(train_step,{x: trainx[batch*100:(batch+1)*100], y_: trainy[batch*100:(batch+1)*100]})
        print(sess.run(accuracy,{x: val_trainx[:], y_: val_trainy[:]}))

    parameters = weights.copy()
    saver = tf.train.Saver(parameters)
    save_path = saver.save(sess, "Weights/models_cnn/model2.ckpt")

    #sess.close()





def test(testX):
    parameters = weights.copy()
    saver = tf.train.Saver(parameters)
    saver.restore(sess,"Weights/models_cnn/model2.ckpt")

    #new_weights = {

    #}

    testx = np.array([testX[idx].reshape((784)).astype(np.float32) for idx in xrange(len(testX))])


    #sess = tf.Session()
    pred = model_build(testx,parameters).eval(session=sess)
    labels = np.array([np.argmax(pre) for pre in pred])


    sess.close()

    return labels
