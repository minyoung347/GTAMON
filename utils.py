import pickle as pkl

import numpy as np

from config import lr, input_feature_len, num_classes


def load_data():

    num_nodes = 5
    
    # N x N
    # bitcoin, seoul, crypto, current, money
    adj_mat = np.array([[0,0,1,0,0],
                        [0,0,1,0,0],
                        [1,1,0,1,0],
                        [0,0,1,0,1],
                        [0,0,0,1,0]])
    
    # Normalized laplacian matrix
    # lapl_mat = adj_mat + np.eye(adj_mat.shape[0])
    # lapl_mat = np.eye(adj_mat.shape[0])
    lapl_mat = adj_mat

    # N x (num_features)
    # num_features = 3 # (search frequency, time remaining, average search hours)
    # average search hours => (24 h = 1). when do users search such words?
    features = np.array([[0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
#     features = np.array([[0.8, 0.4, 0.9],
#                          [0.1, 0.1, 0.25],
#                          [0.8, 0.5, 0.84],
#                          [0.85, 0.3, 0.3],
#                          [0.92, 0.5, 0.1]])
    
    labels = np.array([[1, 0],
                       [1, 0],
                       [0, 1],
                       [0, 1],
                       [1, 0]])
#     labels = np.array([[1, 0],
#                        [0, 1],
#                        [1, 0],
#                        [1, 0],
#                        [0, 1]])

    return lapl_mat, features, labels

def load_data_2():

    num_nodes = 5
    
    # N x N
    # bitcoin, seoul, crypto, current, money
    adj_mat = np.array([[0,0,1,0],
                        [0,0,1,0],
                        [1,1,0,1],
                        [0,0,1,0]])
    
    # Normalized laplacian matrix
    # lapl_mat = adj_mat + np.eye(adj_mat.shape[0])
    # lapl_mat = np.eye(adj_mat.shape[0])
    lapl_mat = adj_mat

    # N x (num_features)
    # num_features = 3 # (search frequency, time remaining, average search hours)
    # average search hours => (24 h = 1). when do users search such words?
    features = np.array([[0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
#     features = np.array([[0.8, 0.4, 0.9],
#                          [0.1, 0.1, 0.25],
#                          [0.8, 0.5, 0.84],
#                          [0.85, 0.3, 0.3]])
    
    labels = np.array([[1, 0],
                       [1, 0],
                       [0, 1],
                       [0, 1]])
#     labels = np.array([[1, 0],
#                        [0, 1],
#                        [1, 0],
#                        [1, 0]])

    return lapl_mat, features, labels

def load_data_3():

    num_nodes = 1000
    
    # N x N
    adj_mat = np.ones((num_nodes, num_nodes))
    
    # Normalized laplacian matrix
    lapl_mat = adj_mat + np.eye(adj_mat.shape[0])
    # lapl_mat = np.eye(adj_mat.shape[0])
    # lapl_mat = adj_mat
    
    features = np.ones((num_nodes, input_feature_len))
    
    labels = np.ones((num_nodes, num_classes))

    return lapl_mat, features, labels

def load_data_cora():

    f = open("cora_data.pkl", "rb")
    adj, features, y_train, y_val, y_test, mask_train, mask_val, mask_test = pkl.load(f)
    f.close()
    
    num_nodes = len(adj)
    lapl_mat = adj + np.eye(num_nodes)
    labels = y_train+y_val+y_test

    return lapl_mat, features, labels, mask_train, mask_val, mask_test