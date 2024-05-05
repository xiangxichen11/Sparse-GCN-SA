import numpy as np
import random
import scipy.sparse as sp
from sklearn.decomposition import PCA
import torch as th

def ratio_homo(adj, labels):
    adjt = adj.t()
    adj = adj+adjt
    n_nodes = adj.size(0)
    index=th.arange(0,n_nodes).long().cpu()
    SUM = 1
    j = 1
    for i in range(n_nodes):
        node_neighbors = adj[i,:]
        indexs_node_neighbors = index[node_neighbors > 0]
        label_node = labels[i]
        label_neighbors = labels[indexs_node_neighbors]
        n_neighbors = label_neighbors.size(0)
        n_neighbors = max(1,n_neighbors)
        if n_neighbors==0:
            continue
        else:
            range_neigibors = th.arange(0, n_neighbors ).long().cpu()
            real_neighbors = range_neigibors[label_neighbors == label_node]
            n_same_neighbor = real_neighbors.size(0)
            j = j + n_same_neighbor-1
            SUM = SUM + n_neighbors-1
    ratio =j/SUM
    return ratio,SUM

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

 # "cora", "citeseer", "pubmed",  "squirrel",  "chameleon", "cornell", "texas",  "wisconsin"
def acquiretvt(dataset, trainsplit,labels,i):
    if dataset=='cora':
        if trainsplit==0.48:
            tvt_index = np.load('./tvtsplit/cora/0.48/' + 'cora_tvt_index' + str(i) + '.npy')
        else:tvt_index = np.load('./tvtsplit/cora/0.6/' + 'cora_tvt_index' + str(i) + '.npy')
    if dataset=='citeseer':
        if trainsplit==0.48:
            tvt_index = np.load('./tvtsplit/citeseer/0.48/' + 'citeseer_tvt_index' + str(i) + '.npy')
        else:tvt_index = np.load('./tvtsplit/citeseer/0.6/' + 'citeseer_tvt_index' + str(i) + '.npy')
    if dataset=='pubmed':
         train_idx, val_idx, test_idx = randomsplit(labels, trainsplit)
         tvt_index = np.array(th.concatenate([train_idx, val_idx, test_idx]))
    if dataset=='squirrel':
        if trainsplit==0.48:
            tvt_index = np.load('./tvtsplit/squirrel/0.48/' + 'squirrel_tvt_index' + str(i) + '.npy')
        else:tvt_index = np.load('./tvtsplit/squirrel/0.6/' + 'squirrel_tvt_index' + str(i) + '.npy')
    if dataset=='chameleon':
        if trainsplit==0.48:
            tvt_index = np.load('./tvtsplit/chameleon/0.48/' + 'chameleon_tvt_index' + str(i) + '.npy')
        else:tvt_index = np.load('./tvtsplit/chameleon/0.6/' + 'chameleon_tvt_index' + str(i) + '.npy')
    if dataset=='cornell':
        if trainsplit==0.48:
            tvt_index = np.load('./tvtsplit/cornell/0.48/' + 'cornell_tvt_index' + str(i) + '.npy')
        else:tvt_index = np.load('./tvtsplit/cornell/0.6/' + 'cornell_tvt_index' + str(i) + '.npy')
    if dataset=='texas':
        if trainsplit==0.48:
            tvt_index = np.load('./tvtsplit/texas/0.48/' + 'texas_tvt_index' + str(i) + '.npy')
        else:tvt_index = np.load('./tvtsplit/texas/0.6/' + 'texas_tvt_index' + str(i) + '.npy')
    if dataset=='wisconsin':
        if trainsplit==0.48:
            tvt_index = np.load('./tvtsplit/wisconsin/0.48/' + 'wisconsin_tvt_index' + str(i) + '.npy')
        else:tvt_index = np.load('./tvtsplit/wisconsin/0.6/' + 'wisconsin_tvt_index' + str(i) + '.npy')
    tvt_index = th.tensor(tvt_index)
    train_n = int(np.ceil(tvt_index.size(0)*trainsplit+th.max(labels)))
    val_n = int(np.ceil(tvt_index.size(0) *(1-trainsplit-0.2)+th.max(labels)))
    idx_train = th.LongTensor(tvt_index[:train_n].numpy())
    idx_val =th.LongTensor( tvt_index[train_n:train_n+val_n].numpy())
    idx_test = th.LongTensor(tvt_index[train_n+val_n:].numpy())
    return idx_train,idx_val,idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def randomsplit(labels, train_percent):
    val_percent = 1-0.2-train_percent
    num_class = th.max(labels) + 1
    one_labels = th.linspace(0, labels.size(0)-1, labels.size(0))
    one_labels = th.as_tensor(one_labels).cpu()
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
        indexs = indexs.cpu().numpy().tolist()
        random.shuffle(indexs)
        indexs = th.Tensor(indexs)
        if a[i] > 1:
            n_train = int(np.ceil(train_percent*a[i]))
            n_val = int(val_percent*a[i])
            train_idx = th.cat((train_idx, indexs[0:n_train]), dim=0)
            val_idx = th.cat((val_idx, indexs[n_train:n_train+ n_val]), dim=0)
            test_idx = th.cat((test_idx, indexs[n_train + n_val:]), dim=0)
        else:
            train_idx = th.cat((train_idx, indexs), dim=0)
            # val_idx = th.cat((val_idx, indexs), dim=0)
            # test_idx = th.cat((test_idx, indexs), dim=0)

        train_idx = th.LongTensor(train_idx.numpy())
        val_idx = th.LongTensor(val_idx.numpy())
        test_idx = th.LongTensor(test_idx.numpy())


    return train_idx, val_idx, test_idx

def pca(features):
    pca = PCA()
    pca.fit(features)

    explained_variance_ratio = pca.explained_variance_ratio_

    deltas = np.diff(explained_variance_ratio)
    optimal_components = np.argmax(deltas) + 1

    pca_optimal = PCA(n_components=optimal_components)
    features = pca_optimal.fit_transform(features)

    return features