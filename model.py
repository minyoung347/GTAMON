import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from config import lr, input_feature_len, num_classes
from config import reddit_num_input, reddit_num_feature, reddit_num_embd_feature, reddit_num_classes


gpu_num = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num


def uniform(shape, half_interval=0.05, name=None):
    init_value = tf.random_uniform(shape, minval=-half_interval, maxval=half_interval, dtype=tf.float32)
    return tf.Variable(init_value, name=name)

def glorot(shape, name=None):
    half_interval = np.sqrt(6.0/(shape[0]+shape[1]))
    return uniform(shape, half_interval, name=name)

def graph_conv_layer(lapl_mat, input_x, feature_shape, name, is_last=False):
    
    with tf.variable_scope(name+"_vars"):
        weight = glorot(feature_shape, name="weight")
        bias = uniform([feature_shape[-1]], name="bias")
        
        out = tf.matmul(lapl_mat, tf.add(tf.matmul(input_x, weight), bias), name="out")
       
        if is_last:
            act = out
            return act
        act = tf.nn.relu(out, name="act")
        return act

def graph_conv_layer_reddit(lapl_mat_list, input_x_list, feature_shape, name, is_last=False):
    
    act_list = []
    with tf.variable_scope(name+"_vars"):
        weight = glorot(feature_shape, name="weight")
        bias = uniform([feature_shape[-1]], name="bias")
        
        for i, (lapl_mat, input_x) in enumerate(zip(lapl_mat_list, input_x_list)):
            out = tf.matmul(lapl_mat, tf.add(tf.matmul(input_x, weight), bias), name="out_{}".format(i))
       
            if is_last:
                act = tf.reduce_mean(out, axis=0, name="gap")
                # act = out
                act_list.append(act)
            else:
                # Global Average Pooling
                act = tf.nn.relu(out, name="act_{}".format(i))
                act_list.append(act)
        return act_list    

def build_model():
    plhdr_input = tf.placeholder(tf.float32, shape=(None, 3), name='input')
    plhdr_label = tf.placeholder(tf.float32, shape=(None, 2), name='label')
    plhdr_lapl = tf.placeholder(tf.float32, shape=(None, None), name='adj')
    
    graph_layer_1 = graph_conv_layer(plhdr_lapl, plhdr_input, [3, 2], name="gc1")
    graph_layer_2 = graph_conv_layer(plhdr_lapl, graph_layer_1, [2, 3], name="gc2")
    graph_layer_3 = graph_conv_layer(plhdr_lapl, graph_layer_2, [3, 2], name="gc3", is_last=True)
    
    # Cross entropy with softmax as loss function
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=graph_layer_3, 
                                                                               labels=plhdr_label), 
                                       name="xent")
    # AdamOptimizer for train network
    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
    # Accuracy
    with tf.name_scope("accuracy"):
        prediction = tf.argmax(graph_layer_3, 1)
        correction = tf.argmax(plhdr_label, 1)
        # Returns the truth value (Integer) of (x == y) element-wise;
        correct_prediction = tf.cast(tf.equal(prediction, correction), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    
    return accuracy, cross_entropy, train
    
def build_model_3():
    plhdr_input = tf.placeholder(tf.float32, shape=(None, input_feature_len), name='input')
    plhdr_label = tf.placeholder(tf.float32, shape=(None, num_classes), name='label')
    plhdr_lapl = tf.placeholder(tf.float32, shape=(None, None), name='adj')
    
    graph_layer_1 = graph_conv_layer(plhdr_lapl, plhdr_input, [input_feature_len, 30], name="gc1")
    graph_layer_2 = graph_conv_layer(plhdr_lapl, graph_layer_1, [30, 30], name="gc2")
    graph_layer_3 = graph_conv_layer(plhdr_lapl, graph_layer_2, [30, num_classes], name="gc3", is_last=True)
    
    # Cross entropy with softmax as loss function
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=graph_layer_3, 
                                                                               labels=plhdr_label), 
                                       name="xent")
    # AdamOptimizer for train network
    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
    # Accuracy
    with tf.name_scope("accuracy"):
        prediction = tf.argmax(graph_layer_3, 1)
        correction = tf.argmax(plhdr_label, 1)
        # Returns the truth value (Integer) of (x == y) element-wise;
        correct_prediction = tf.cast(tf.equal(prediction, correction), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    
    return accuracy, cross_entropy, train

def build_model_cora():
    plhdr_mask = tf.placeholder(tf.float32, shape=(None,), name="mask")
    plhdr_input = tf.placeholder(tf.float32, shape=(None, 1433), name='input')
    plhdr_label = tf.placeholder(tf.float32, shape=(None, 7), name='label')
    plhdr_lapl = tf.placeholder(tf.float32, shape=(None, None), name='adj')
    
    graph_layer_1 = graph_conv_layer(plhdr_lapl, plhdr_input, [1433, 1000], name="gc1")
    graph_layer_2 = graph_conv_layer(plhdr_lapl, graph_layer_1, [1000, 1000], name="gc2")
    graph_layer_3 = graph_conv_layer(plhdr_lapl, graph_layer_2, [1000, 7], name="gc3", is_last=True)
    
    # Cross entropy with softmax as loss function
    with tf.name_scope("xent"):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=graph_layer_3,labels=plhdr_label)
        
        mask = tf.cast(plhdr_mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        
        cross_entropy = tf.reduce_mean(loss, name="xent")
        
    # AdamOptimizer for train network
    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(3e-6).minimize(cross_entropy)
        
    # Accuracy
    with tf.name_scope("accuracy"):
        prediction = tf.argmax(graph_layer_3, 1)
        correction = tf.argmax(plhdr_label, 1)
        # Returns the truth value (Integer) of (x == y) element-wise;
        correct_prediction = tf.cast(tf.equal(prediction, correction), tf.float32)
        correct_prediction *= mask  
        accuracy = tf.reduce_mean(correct_prediction, axis=0)
    
    return accuracy, cross_entropy, train
    
def build_model_reddit():
    
    plhdr_input_list = [tf.placeholder(tf.float32, 
                                       shape=(None, reddit_num_feature), 
                                       name='input_{}'.format(i)) for i in range(reddit_num_input)]
    plhdr_lapl_list = [tf.placeholder(tf.float32, 
                                      shape=(None, None), 
                                      name='adj_{}'.format(i)) for i in range(reddit_num_input)]
    
    graph_layer_1 = graph_conv_layer_reddit(plhdr_lapl_list, plhdr_input_list, [reddit_num_feature, 500], name="gc1")
    graph_layer_2 = graph_conv_layer_reddit(plhdr_lapl_list, graph_layer_1, [500, 500], name="gc2")
    graph_layer_3 = graph_conv_layer_reddit(plhdr_lapl_list, graph_layer_2, [500, reddit_num_embd_feature], name="gc3", is_last=True)
    
    graph_layer_3_reshaped = tf.reshape(graph_layer_3, (reddit_num_input, reddit_num_embd_feature), name="gc3_reshaped")
    
    return graph_layer_3_reshaped

def build_rnn_reddit(gcn_output):
    
    # The shape of 'gcn_outout' is same with that of 'graph_layer_3_reshaped'
    
    input_sequence = tf.expand_dims(gcn_output, axis=0, name="rnn_input")
    
    hidden_size = 3
    w = tf.Variable(tf.truncated_normal([hidden_size, reddit_num_classes]))
    b = tf.Variable(tf.truncated_normal([reddit_num_classes]))
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, input_sequence, dtype=tf.float32)
    logits = tf.add(tf.matmul(outputs[:,-1], w), b, name='logits')
    
    return logits