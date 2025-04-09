from __future__ import print_function

import torch
import warnings
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import logging
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F


warnings.filterwarnings(action='once')

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def proto_align_loss(feat, feat_aug, temperature=0.3):
    cl_dim = feat.shape[0]

    feat_norm = torch.norm(feat, dim=-1)
    feat = torch.div(feat, feat_norm.unsqueeze(1))

    feat_aug_norm = torch.norm(feat_aug, dim=-1)
    feat_aug = torch.div(feat_aug, feat_aug_norm.unsqueeze(1))

    sim_clean = torch.mm(feat, feat.t())
    mask = (torch.ones_like(sim_clean) - torch.eye(cl_dim, device=sim_clean.device)).bool()
    sim_clean = sim_clean.masked_select(mask).view(cl_dim, -1)

    sim_aug = torch.mm(feat, feat_aug.t())
    sim_aug = sim_aug.masked_select(mask).view(cl_dim, -1)

    logits_pos = torch.bmm(feat.view(cl_dim, 1, -1), feat_aug.view(cl_dim, -1, 1)).squeeze(-1)
    logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

    logits = torch.cat([logits_pos, logits_neg], dim=1)
    instance_labels = torch.zeros(cl_dim).long().to(sim_clean.device)

    loss = F.cross_entropy(logits / temperature, instance_labels)

    return loss

def CEloss(y_true, y_pred, num_classes):
    y_true = torch.nn.functional.one_hot(y_true, num_classes=num_classes).float()
    return nn.functional.cross_entropy(y_pred, y_true.argmax(dim=1))


def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = sp.csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def process_adj(adj):
    '添加自环、对称化以及归一化，并转化为gnn的合适的输入'
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def get_knn_graph(x, num_neighbor, batch_size=0, knn_metric='cosine', connected_fast=True):
    if not batch_size:
        adj_knn = kneighbors_graph(x, num_neighbor, metric=knn_metric)
    else:
        if connected_fast:
            print('compute connected fast knn')
            num_neighbor1 = int(num_neighbor / 2)
            batches1 = get_random_batch(x.shape[0], batch_size)
            row1, col1 = global_knn(x, num_neighbor1, batches1, knn_metric)
            num_neighbor2 = num_neighbor - num_neighbor1
            batches2 = get_random_batch(x.shape[0], batch_size)
            row2, col2 = global_knn(x, num_neighbor2, batches2, knn_metric)
            row, col = np.concatenate((row1, row2)), np.concatenate((col1, col2))
        else:
            print('compute fast knn')
            batches = get_random_batch(x.shape[0], batch_size)
            row, col = global_knn(x, num_neighbor, batches, knn_metric)
        adj_knn = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))

    return adj_knn.tocoo()  # .tolil()

def global_knn(x, num_neighbor, batches, knn_metric):
    row = None
    for batch in batches:
        knn_current = kneighbors_graph(x[batch], num_neighbor, metric=knn_metric).tocoo() # tocoo稀疏矩阵
        row_current = batch[knn_current.row]
        col_current = batch[knn_current.col]
        if row is None:
            row = row_current
            col = col_current
        else:
            row = np.concatenate((row, row_current))
            col = np.concatenate((col, col_current))
    return row, col

def get_random_batch(n, batch_size):
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    batches = []
    i = 0
    while i + batch_size * 2 < n:
        batches.append(idxs[i:i + batch_size])
        i += batch_size
    batches.append(idxs[i:])
    return batches

def feature_propagation(adj, features, K, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj = adj.to(device)
    features_prop = features.clone()
    for i in range(1, K + 1):
        features_prop = torch.sparse.mm(adj, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
    return features_prop.cpu()


def plot_confusion_matrix(cm, label, confusion_matrix_path):
    logging.disable(logging.CRITICAL)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index=[i for i in label], columns=[i for i in label])
    sns.heatmap(df_cm, annot=True, fmt='.2f')
    fig = plt.gcf()
    plt.subplots_adjust(bottom=0.3, left=0.2)
    fig.savefig(confusion_matrix_path, dpi=100)
    plt.close()


def plot_recall(labels, recall, recall_path):
    logging.disable(logging.CRITICAL)

    # create data
    x_pos = np.arange(0, len(labels) * 2, 2)
    _recall = np.round(recall * 100, 2)

    # create bars
    plt.bar(x_pos, _recall, width=0.5)

    # rotation of the bar names
    plt.xticks(x_pos, labels, rotation=70)
    # custom the subplot layout
    plt.subplots_adjust(bottom=0.3, top=0.8)
    # enable grid
    plt.grid(True)

    plt.title('Detection Rate')
    plt.ylabel('recall score')

    # print value on the top of bar
    x_locs, x_labs = plt.xticks()
    for i, v in enumerate(_recall):
        plt.text(x_locs[i] - 0.6, v + 5, str(v))

    # set limit on y label
    plt.ylim(0, max(_recall) + 15)

    # savefig
    fig = plt.gcf()
    fig.savefig(recall_path, dpi=100)
    plt.close()

    logging.disable(logging.NOTSET)

def create_directory(p):
    from pathlib import Path
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    return logger


class Z_Scaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit_transform(self, data):

        self.mean = data.mean()
        self.std = data.std()

        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std

        return (data - mean) / std

class Sigmoid_Scaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit_transform(self, data):
        data = np.array(data)
        return 1.0 / (1 + np.exp(-float(data)))
