import numpy as np
import scipy.sparse as sp
import torch

from sklearn.metrics import f1_score, accuracy_score

def encode_onehot(labels):
    classes = list(set(labels))
    classes.sort()
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot, classes_dict


#def load_data(path="./data/cora/", dataset="cora"):
#    """Load citation network dataset (cora only for now)"""
#    print('Loading {} dataset...'.format(dataset))
#
#    import pdb
#    pdb.set_trace()
#    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
#    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#    labels = encode_onehot(idx_features_labels[:, -1])
#
#    # build graph
#    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#    idx_map = {j: i for i, j in enumerate(idx)}
#    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
#    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
#
#    # build symmetric adjacency matrix
#    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#    features = normalize_features(features)
#    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
#
#    idx_train = range(3600)
#    idx_val = range(3601, 4000)
#    idx_test = range(4100, 4100)
#
#    adj = torch.FloatTensor(np.array(adj.todense()))
#    features = torch.FloatTensor(np.array(features.todense()))
#    labels = torch.LongTensor(np.where(labels)[1])
#
#    idx_train = torch.LongTensor(idx_train)
#    idx_val = torch.LongTensor(idx_val)
#    idx_test = torch.LongTensor(idx_test)
#
#    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_text_gnn(path, dataset=None):
    print('Loading data from {}...'.format(path))

    # load doc nodes
    doc_idx_features_labels = np.genfromtxt("{}/doc.nodes".format(path), dtype=np.dtype(str))
    doc_features = np.asarray(doc_idx_features_labels[:, 1:-1], dtype=np.float32)
    labels, classes_dict = encode_onehot(doc_idx_features_labels[:, -1])
    doc_idx = np.array(doc_idx_features_labels[:, 0], dtype=np.int32)
    print('Load doc nodes: {}'.format(len(doc_idx)))
    print('Encoded classes: {}'.format(classes_dict))

    # load word nodes
    word_idx_features_labels = np.genfromtxt("{}/word.nodes".format(path), dtype=np.dtype(str))
    word_features = np.asarray(word_idx_features_labels[:, 1:], dtype=np.float32)
    word_idx = np.array(word_idx_features_labels[:, 0], dtype=np.int32)
    print('Load word nodes: {}'.format(len(word_idx)))

    features = np.vstack((doc_features, word_features))

    #import pdb
    #pdb.set_trace()
    # build graph
    idx = doc_idx.tolist() + word_idx.tolist()
    node_size = len(idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    weighted_edges = np.genfromtxt("{}/data.edges".format(path), dtype=np.dtype(str))
    edges_unordered = np.asarray(weighted_edges[:, 0:2], dtype=np.int32)
    weights = np.asarray(weighted_edges[:, 2], dtype=np.float32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(node_size, node_size), dtype=np.float32)

    # build symmetric adjacency matrix
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = adj + np.identity(node_size)
    #adj = normalize_adj(adj)
    #features = normalize_features(features)

    # Load data split index
    #data_split = np.genfromtxt("{}/data.split".format(path), dtype=np.int32)
    #train_size = data_split[0]
    #dev_size = data_split[1]
    #test_size = data_split[2]

    doc_size = len(doc_idx)
    train_size = int(doc_size * 0.9)
    idx_train = range(train_size)
    idx_val = range(train_size, doc_size)
    idx_test = range(train_size, doc_size)

    #idx_train = range(140)
    #idx_val = range(200, 500)
    #idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    #adj = torch.FloatTensor(np.array(adj))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def pos_class_f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    score = f1_score(y_true=labels.cpu(), y_pred=preds.cpu())
    #score = f1_score(y_true=labels, y_pred=preds)
    return float(score)

