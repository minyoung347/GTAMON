import time

import numpy as np
import tensorflow as tf

from utils import load_data, load_data_2, load_data_3
from model import build_model, build_model_3

from config import epochs, print_period


lapl_mat, features, labels = load_data()
lapl_mat_2, features_2, labels_2 = load_data_2()

lapl_mat_3, features_3, labels_3 = load_data_3()
print("lapl_mat", lapl_mat.shape)
print("features", features.shape)
print("labels", labels.shape)

tf_config = tf.ConfigProto()
graph = tf.Graph()
with graph.as_default():
    accuracy, cross_entropy, train = build_model()
    # accuracy, cross_entropy, train = build_model_3()
    # saver = tf.train.Saver(max_to_keep=10)

def print_acc_loss(sess, ops, feed_dict, epoch):
    print(epoch, sess.run(ops, feed_dict=feed_dict))
    
def run_train(sess, ops, feed_dict):
    sess.run(ops, feed_dict=feed_dict)

with tf.Session(graph=graph, config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    
    ops = [accuracy, cross_entropy]
    feed_dict = {'input:0': features, 'label:0': labels, 'adj:0': lapl_mat}
    feed_dict_2 = {'input:0': features_2, 'label:0': labels_2, 'adj:0': lapl_mat_2}
    feed_dict_3 = {'input:0': features_3, 'label:0': labels_3, 'adj:0': lapl_mat_3}
    
    
    print_acc_loss(sess, ops, feed_dict, 0)
    print_acc_loss(sess, ops, feed_dict_2, 0)
    # print_acc_loss(sess, ops, feed_dict_3, 0)
    
    i_time = time.time()
    for i in range(epochs):
        run_train(sess, [train], feed_dict)
        run_train(sess, [train], feed_dict_2)
        # run_train(sess, [train], feed_dict_3)
        
        if i%print_period == 0:
            print("="*80)
            print_acc_loss(sess, ops, feed_dict, i+1)
            print_acc_loss(sess, ops, feed_dict_2, i+1)
            # print_acc_loss(sess, ops, feed_dict_3, i+1)
            
            print("Time - {} s".format(time.time()-i_time))
            i_time = time.time()