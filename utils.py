import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx
import collections
from collections import defaultdict
import torch
import os
import random
import torch.nn.functional as F


def load_data_new(dataset_str, split):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # print('dataset_str', dataset_str)
    # print('split', split)
    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = list(np.where(train_mask == 1)[0])
        idx_val = list(np.where(val_mask == 1)[0])
        idx_test = list(np.where(test_mask == 1)[0])

        no_label_nodes = []
        if dataset_str == 'citeseer':  # citeseer has some data with no label
            for i in range(len(labels)):
                if sum(labels[i]) < 1:
                    labels[i][0] = 1
                    no_label_nodes.append(i)

            for n in no_label_nodes:  # remove unlabel nodes from train/val/test
                if n in idx_train:
                    idx_train.remove(n)
                if n in idx_val:
                    idx_val.remove(n)
                if n in idx_test:
                    idx_test.remove(n)

    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_str,
                                                                f'out1_node_feature_label.txt')
        graph_dict = defaultdict(list)
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                graph_dict[int(line[0])].append(int(line[1]))
                graph_dict[int(line[1])].append(int(line[0]))

        # print(sorted(graph_dict))
        graph_dict_ordered = defaultdict(list)
        for key in sorted(graph_dict):
            graph_dict_ordered[key] = graph_dict[key]
            graph_dict_ordered[key].sort()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))
        # adj = sp.csr_matrix(adj)

        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features_list = []
        for key in sorted(graph_node_features_dict):
            features_list.append(graph_node_features_dict[key])
        features = np.vstack(features_list)
        features = sp.csr_matrix(features)

        labels_list = []
        for key in sorted(graph_labels_dict):
            labels_list.append(graph_labels_dict[key])

        label_classes = max(labels_list) + 1
        labels = np.eye(label_classes)[labels_list]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]

    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels))[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # print('adj', adj.shape)
    # print('features', features.shape)
    # print('labels', labels.shape)
    # print('idx_train', idx_train.shape)
    # print('idx_val', idx_val.shape)
    # print('idx_test', idx_test.shape)
    return adj, features, labels, idx_train, idx_val, idx_test


# Read split data
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def add_noise(pre_noise, labels, idx_clean, idx_unknown, noise_type):
    num_classes = labels.max().item() + 1
    if noise_type == 'uniform':
        c = pre_noise * np.full((num_classes, num_classes), 1 / num_classes) + \
            (1 - pre_noise) * np.eye(num_classes)
    else:
        seed = 1
        np.random.seed(seed)
        c = np.eye(num_classes) * (1 - pre_noise)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            c[i][np.random.choice(row_indices[row_indices != i])] = pre_noise

    y_unkonwn = labels[idx_unknown].clone() # the labels of training set

    for i in range(len(idx_unknown)):   # add noise
        y_unkonwn[i] = np.random.choice(num_classes, p=c[y_unkonwn[i]])
    new_labels = labels.clone()
    new_labels[idx_unknown] = y_unkonwn  # the noisy labels

    c = (new_labels.max() + 1)  # class numbers
    n = new_labels.shape[0]  #  node numbers
    y_clean = torch.zeros((n, c))
    y_unknown = torch.zeros((n, c))
    
    y_clean[idx_clean] = F.one_hot(new_labels[idx_clean], c).double().squeeze(1) # the matrix of clean labels
    y_unknown[idx_unknown] = F.one_hot(new_labels[idx_unknown], c).double().squeeze(1)  # the matrix of noisy labels

    return new_labels, y_clean, y_unknown

def reset_idx(idx_train, idx_val, idx_test, pre_clean, pre_unknown):
    idx_all = torch.cat((idx_train, idx_val, idx_test))

    num_clean = int(idx_all.shape[0]*pre_clean)
    idx_clean = idx_all[:num_clean]

    num_unknown = int(idx_all.shape[0]*pre_unknown)
    idx_unknown = idx_all[num_clean:(num_clean+num_unknown)]

    num_val = int((idx_all.shape[0] - num_clean - num_unknown)*0.5)
    idx_val = idx_all[(num_clean+num_unknown):(num_clean+num_unknown+num_val)]

    idx_test = idx_all[(num_clean+num_unknown+num_val):]
    
    return idx_clean, idx_unknown, idx_val, idx_test
