import networkx as nx
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
import yaml
import os

'''
Copyright (c) 2021 Weihua Hu
Released under the MIT license
https://github.com/weihua916/powerful-gnns/blob/master/LICENSE
'''

# https://github.com/weihua916/powerful-gnns/blob/master/util.py


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of
                           the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list,
                      will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, label = [int(w) for w in row]
            if not label in label_dict:
                mapped = len(label_dict)
                label_dict[label] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array(
                        [float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                # node_feature_flag = True
            else:
                node_features = None
                # node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, label, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        # deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag]
                                                  for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def get_y_data(graph):
    return graph.label


class EarlyStopping:
    def __init__(self, patience=30, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_score_prev = np.Inf
        self.model = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_score_prev = val_loss
            self.model = copy.deepcopy(model)
            return True
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = val_loss
            if self.verbose:
                print(
                    f'Validation loss decreased ({self.best_score_prev:.6f} --> {val_loss:.6f}).')

            self.best_score_prev = val_loss
            self.counter = 0
            self.model = copy.deepcopy(model)
            return True


class HoldPerformanceIndices:
    '''
    Class for scores
    '''

    def __init__(self):
        self.best_train_acc = 0
        self.best_val_acc = 0
        self.best_train_loss = np.Inf
        self.best_val_loss = np.Inf
        self.maxnum_epoch = 0
        self.train_acc_list = []
        self.val_acc_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.epoch_list = []

    def __call__(self, epoch, train_acc, val_acc, train_loss,
                 val_loss, best_score):
        '''
        Update scores.
        Args.
            epoch(int): Current number of epoch.
            train_acc(float): Current train accuracy.
            val_acc(float): Current valid accuracy.
            train_loss(float): Current train loss.
            val_loss(float): Current valid loss.
            best_score(bool): Update scores as best score.
        '''
        self.train_acc_list.append(train_acc)
        self.val_acc_list.append(val_acc)
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)
        self.epoch_list.append(epoch)
        if best_score:
            self.best_train_acc = train_acc
            self.best_val_acc = val_acc
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.maxnum_epoch = epoch


def output_board(hold_performance_indices, log_dir):
    '''
    Args.
        hold_performanceindices(HoldPerformanceIndices): results
        log_dir(Path): output log file
    '''
    writer = SummaryWriter(log_dir=log_dir)
    output_params = zip(hold_performance_indices.epoch_list,
                        hold_performance_indices.train_acc_list,
                        hold_performance_indices.val_acc_list,
                        hold_performance_indices.train_loss_list,
                        hold_performance_indices.val_loss_list
                        )
    for epoch, train_acc, val_acc, train_loss, val_loss in output_params:
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

    writer.close()


def get_params(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
