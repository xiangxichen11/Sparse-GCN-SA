import os
import random
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
import torch as th
from data_prop import sparse_mx_to_torch_sparse_tensor

def load_data(dataset_name, train_percentage=None):
    val_percentage = 1-train_percentage-0.2
    graph_adjacency_list_file_path = os.path.join('..', 'new_data', dataset_name, 'out1_graph_edges.txt')  # 路径拼接
    graph_node_features_and_labels_file_path = os.path.join('..', 'new_data', dataset_name,
                                                            f'out1_node_feature_label.txt')
    G = nx.DiGraph()  # 建立一个有向图
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')  # 每行包含三个元素，‘idx' 'feature' 'label'
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','),
                                                                  dtype=np.uint8)  # 按照','把特征字符串进行分割
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:  # 构图
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))  # 邻接矩阵

    features = np.array(
        [features for _, features in
         sorted(G.nodes(data='features'), key=lambda x: x[0])])  # key=lambda x: x是固定用法，0代表按第一个元素排序，
    
    #Included PCA to reduce features
    features = _pca(features)

    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    assert (train_percentage is not None and val_percentage is not None)
    assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

    labels = th.tensor(labels, dtype=th.int8)
    [idx_train, idx_val, idx_test] = randomsplit(labels, train_percentage, val_percentage)

    idx_train = th.as_tensor(idx_train, dtype=int)
    idx_val = th.as_tensor(idx_val, dtype=int)
    idx_test = th.as_tensor(idx_test, dtype=int)

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.as_tensor(labels, dtype=int)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj1 = adj + sp.eye(adj.shape[0])
    adj1 = normalize_adj(adj1)
    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)

    features = features.cpu()
    adj1 = adj1.cpu()
    labels = labels.cpu()
    # idx_train = idx_train.cpu()
    # idx_val = idx_val.cpu()
    # idx_test = idx_test.cpu()


    return adj1, features, labels
def _pca(features):
    pca = PCA()
    pca.fit(features)

    explained_variance_ratio = pca.explained_variance_ratio_
    optimal_components = _elbow_method(explained_variance_ratio)
    
    pca_optimal = PCA(n_components=optimal_components)
    features = pca_optimal.fit_transform(features)
    return features

def _elbow_method(variance_ratio):
    deltas = np.diff(variance_ratio)
    max_delta_index = np.argmax(deltas)
    return max_delta_index + 1

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def randomsplit(labels, train_percent, val_percent):

    num_class = th.max(labels) + 1
    one_labels = th.linspace(0, labels.size(0)-1, labels.size(0))
    one_labels = th.as_tensor(one_labels)
    a = th.zeros(num_class, 1)
    for i in range(num_class):
        b = labels == i
        a[i] = (one_labels[b]).size(0)

    # print(a)
    train_idx = []
    train_idx = th.tensor(train_idx)
    val_idx = test_idx = train_idx

    for i in range(num_class):
        b = labels == i
        indexs = one_labels[b]
        indexs = indexs.numpy().tolist()
        random.shuffle(indexs)
        indexs = th.Tensor(indexs)
        if a[i] > 1:
            n_train = int(np.ceil(train_percent*a[i]))
            n_val = int(val_percent*a[i])
            train_idx = th.cat((train_idx, indexs[0:n_train]), dim=0)
            val_idx = th.cat((val_idx, indexs[n_train:n_train + n_val]), dim=0)
            test_idx = th.cat((test_idx, indexs[n_train + n_val:]), dim=0)

        else:
            train_idx = th.cat((train_idx, indexs), dim=0)
            val_idx = th.cat((val_idx, indexs), dim=0)
            test_idx = th.cat((test_idx, indexs), dim=0)

        # test_index = th.LongTensor(test_index)
        # train_index = th.LongTensor(train_index)
        # val_index = th.LongTensor(val_index)



    return train_idx, val_idx, test_idx
