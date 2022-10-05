
import copy
import functools
import itertools
import math
import random
from ctypes import c_double, c_int, c_int32

import networkx as nx
import numpy as np
import torch
from numpy.ctypeslib import ndpointer


def insert_noise_use_C_use_map(g_list, alpha, seed, apply_noise,
                               degree_as_tag, epsilon=0):
    '''
    Processing by type of noise
    Args:
        g_list(list(S2VGraph)): Graph classes.
        alpha(float):the proportion of non-private users.
        seed(int): Seed number.
        degree_as_tag(0 or 1): Whether to set the degree of the gragh to the feature. 
        apply_noise:Types of noises.
        epsilon(float):privacy budget.
    '''
    new_dataset_list = []
    if apply_noise == 'baseline':
        new_dataset_list = list(
            map(functools.partial(do_map_baseline1_remove_user,
                                  toc_alpha=alpha, toc_seed=seed,
                                  degree_as_tag=degree_as_tag), g_list))
    elif apply_noise == 'lap':
        new_dataset_list = list(
            map(functools.partial(do_map_LAPGraph_use_alpha, toc_alpha=alpha,
                                  toc_epsilon=epsilon, toc_seed=seed,
                                  degree_as_tag=degree_as_tag), g_list))
    elif apply_noise == 'RR':
        u_lib = np.ctypeslib.load_library(
            "insert_noise_nonprivate_select_rand.so", ".")
        _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
        u_lib.insert_noise_RR_select_rand.argtypes = [
            _DOUBLE_PP, _DOUBLE_PP, c_int32, c_double, c_double, c_int32]
        u_lib.insert_noise_RR_select_rand.restype = None
        u_tp = np.uintp
        graph_length = len(g_list)
        graph_indices = list(range(1, graph_length+1))
        new_dataset_list = list(map(functools.partial(do_map_rr,
                                                      graph_count=graph_length,
                                                      lib=u_lib,
                                tp=u_tp, toc_epsilon=epsilon, toc_alpha=alpha,
                                toc_seed=seed, degree_as_tag=degree_as_tag),
                                g_list, graph_indices))
    elif apply_noise == 'DPRR':
        u_lib = np.ctypeslib.load_library(
            "insert_noise_nonprivate_select_rand.so", ".")
        u_lib_DPRR = np.ctypeslib.load_library(
            "insert_noise_onlyDPRR_select_rand_final.so", ".")
        _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
        u_lib.insert_noise_RR_select_rand.argtypes = [
            _DOUBLE_PP, _DOUBLE_PP, c_int32, c_double, c_double, c_int32]
        u_lib_DPRR.insert_noise_DPRR_select_rand.argtypes = [
            _DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, c_int32, c_double, c_double,
            c_int32, c_int32]
        u_lib.insert_noise_RR_select_rand.restype = None
        u_lib_DPRR.insert_noise_DPRR_select_rand.restype = None
        u_tp = np.uintp
        graph_length = len(g_list)
        graph_indices = list(range(1, graph_length+1))
        new_dataset_list = list(map(functools.partial(do_map_drpp,
                                                      graph_count=graph_length,
                                                      lib=u_lib,
                                                      lib_DPRR=u_lib_DPRR,
                                                      tp=u_tp,
                                                      toc_epsilon=epsilon,
                                                      toc_alpha=alpha,
                                toc_seed=seed, degree_as_tag=degree_as_tag),
                                g_list, graph_indices))
    return new_dataset_list


def do_map_rr(graph, graph_index, graph_count, lib, tp, toc_epsilon, toc_alpha,
              toc_seed, degree_as_tag):
    # Create seeds
    use_seed_RR = graph_index + (toc_seed * graph_count)
    # Create epsilon and interface for computation.
    u_toc_epsilon_normal = c_double(toc_epsilon)
    u_toc_alpha = c_double(toc_alpha)
    u_toc_RR_seed = c_int(use_seed_RR)
    # Create adjacency matrix
    n = nx.number_of_nodes(graph.g)
    adjacency = nx.to_numpy_array(graph.g, nodelist=range(n), weight=None)
    use_adjency = copy.deepcopy(adjacency)
    use_adjency = use_adjency.astype(np.float64)
    # Compute degrees.
    degrees = list(np.sum(use_adjency, axis=1))
    degrees = np.array([degrees])
    degrees = degrees.astype(np.float64)
    # Setting non-private.
    alpha_node_count = math.floor(n * toc_alpha)
    node_index = [i for i in range(n)]
    random.seed(toc_seed)
    non_private = random.sample(node_index, alpha_node_count)
    # Setting private user.
    private = [i for i in node_index if i not in non_private]
    private = np.array([private])
    private = private.astype(np.float64)
    non_private = np.array([non_private])
    non_private = non_private.astype(np.float64)
    toc_n = c_int(n)
    mpp = (use_adjency.__array_interface__[
           'data'][0] + np.arange(use_adjency.shape[0])*use_adjency.strides[0]
           ).astype(tp)
    mpp_nonpra = (non_private.__array_interface__[
                  'data'][0] +
                  np.arange(non_private.shape[0])*non_private.strides[0]
                  ).astype(tp)
    lib.insert_noise_RR_select_rand(
        mpp, mpp_nonpra, toc_n, u_toc_epsilon_normal, u_toc_alpha,
        u_toc_RR_seed)
    # Update neighbors
    neighbors = list(map(get_neighbors, use_adjency))
    graph.neighbors = neighbors
    # Update edge mat
    target_edge_mat = np.argwhere(use_adjency)
    edge_list = torch.LongTensor(target_edge_mat).transpose(0, 1)
    if len(edge_list) != 0:
        graph.edge_mat = edge_list
    else:
        graph.edge_mat = torch.LongTensor([[], []])
    # Create lists to examine adjacent nodes.
    new_degrees = [len(graph.neighbors[j]) for j in range(len(graph.g))]
    graph.max_neighbor = max(new_degrees)
    if degree_as_tag == 1:
        graph.node_tags = new_degrees

    return graph


def do_map_drpp(graph, graph_index, graph_count, lib, lib_DPRR, tp,
                toc_epsilon, toc_alpha, toc_seed, degree_as_tag):
    # Create seed
    use_seed_RR = graph_index + (toc_seed * graph_count)
    use_seed_DPRR = use_seed_RR * -1
    # Create epsilon and interface for computation.
    epsilon_two = (toc_epsilon * 9) / 10
    u_toc_epsilon = c_double(epsilon_two)
    u_toc_epsilon_normal = c_double(toc_epsilon)
    u_toc_alpha = c_double(toc_alpha)
    u_toc_RR_seed = c_int(use_seed_RR)
    u_toc_DPRR_seed = c_int(use_seed_DPRR)
    # Create adjacency matrix
    n = nx.number_of_nodes(graph.g)
    adjacency = nx.to_numpy_array(graph.g, nodelist=range(n), weight=None)
    use_adjency = copy.deepcopy(adjacency)
    use_adjency = use_adjency.astype(np.float64)
    # Compute degrees.
    degrees = list(np.sum(use_adjency, axis=1))
    degrees = np.array([degrees])
    degrees = degrees.astype(np.float64)
    # Setting non-private.
    alpha_node_count = math.floor(n * toc_alpha)
    node_index = [i for i in range(n)]
    random.seed(toc_seed)
    non_private = random.sample(node_index, alpha_node_count)
    # Setting private user.
    private = [i for i in node_index if i not in non_private]
    u_toc_private_count = c_int(len(private))
    private = np.array([private])
    private = private.astype(np.float64)
    non_private = np.array([non_private])
    non_private = non_private.astype(np.float64)
    toc_n = c_int(n)
    mpp = (use_adjency.__array_interface__[
           'data'][0] +
           np.arange(use_adjency.shape[0])*use_adjency.strides[0]).astype(tp)
    mpp_nonpra = (non_private.__array_interface__[
                  'data'][0] +
                  np.arange(non_private.shape[0])*non_private.strides[0]
                  ).astype(tp)
    mpp_private = (private.__array_interface__[
                   'data'][0] +
                   np.arange(private.shape[0])*private.strides[0]).astype(tp)
    mpp_degrees = (degrees.__array_interface__[
                   'data'][0] +
                   np.arange(degrees.shape[0])*degrees.strides[0]).astype(tp)
    lib.insert_noise_RR_select_rand(
        mpp, mpp_nonpra, toc_n, u_toc_epsilon, u_toc_alpha, u_toc_RR_seed)
    # Start DPRR
    lib_DPRR.insert_noise_DPRR_select_rand(
        mpp, mpp_private, mpp_degrees, toc_n, u_toc_epsilon_normal,
        u_toc_alpha, u_toc_private_count, u_toc_DPRR_seed)

    # Update neighbors.
    neighbors = list(map(get_neighbors, use_adjency))
    graph.neighbors = neighbors
    # Update edge_mat.
    target_edge_mat = np.argwhere(use_adjency)
    edge_list = torch.LongTensor(target_edge_mat).transpose(0, 1)
    if len(edge_list) != 0:
        graph.edge_mat = edge_list
    else:
        graph.edge_mat = torch.LongTensor([[], []])
    # Create lists to examine adjacent nodes.
    new_degrees = [len(graph.neighbors[j]) for j in range(len(graph.g))]
    graph.max_neighbor = max(new_degrees)
    if degree_as_tag == 1:
        graph.node_tags = new_degrees
    return graph


def do_map_baseline1_remove_user(graph, toc_alpha, toc_seed, degree_as_tag):
    n = nx.number_of_nodes(graph.g)
    adjacency = nx.to_numpy_array(graph.g, nodelist=range(n), weight=None)
    use_adjency = copy.deepcopy(adjacency)
    delete_localuser_node_count = math.floor(n * (1 - toc_alpha))
    # Set -1 for all nodes that do not add noise.
    node_index = [i for i in range(n)]
    random.seed(toc_seed)
    delete_localuser = random.sample(node_index, delete_localuser_node_count)
    # Delete unused users.
    use_adjency = np.delete(use_adjency, delete_localuser, 0)
    use_adjency = np.delete(use_adjency, delete_localuser, 1)
    graph.g.remove_nodes_from(delete_localuser)
    after_n = nx.number_of_nodes(graph.g)
    neighbors = list(map(get_neighbors, use_adjency))
    graph.neighbors = neighbors
    # Update edge mat.
    target_edge_mat = np.argwhere(use_adjency)
    edge_list = torch.LongTensor(target_edge_mat).transpose(0, 1)
    if len(edge_list) != 0:
        graph.edge_mat = edge_list
    else:
        graph.edge_mat = torch.LongTensor([[], []])
    new_degrees = [len(graph.neighbors[j]) for j in range(len(graph.g))]
    graph.max_neighbor = max(new_degrees)
    # Exist only degree_as_tag = 0.
    if degree_as_tag == 1:
        new_degrees = np.sum(use_adjency, axis=1)
        new_degrees = new_degrees.astype(np.int64)
        new_datax = torch.zeros_like(graph.x).numpy()
        for j in range(n):
            target_degree = new_degrees[j]
            new_datax[j, target_degree] = 1.0
        new_datax = torch.from_numpy(new_datax)
        graph.node_tags = new_datax
    else:
        node_features = torch.ones(after_n, 1)
        graph.node_tags = [0] * after_n
        graph.node_features = node_features
    return graph


def do_map_LAPGraph_use_alpha(graph, toc_alpha, toc_epsilon, toc_seed,
                              degree_as_tag):
    n = nx.number_of_nodes(graph.g)
    adjacency = nx.to_numpy_array(graph.g, nodelist=range(n), weight=None)
    use_adjency = copy.deepcopy(adjacency)
    # Computation of adjacency matrix
    degrees = list(np.sum(use_adjency, axis=1))
    aij = copy.deepcopy(use_adjency)
    output_use_adjency = copy.deepcopy(use_adjency)
    output_use_adjency = np.zeros_like(output_use_adjency)
    flip_targets = np.zeros((n, n))
    # Setting non-private.
    alpha_node_count = math.floor(n * toc_alpha)
    # Set -1 for all nodes that do not add noise.
    node_index = [i for i in range(n)]
    random.seed(toc_seed)
    non_private = random.sample(node_index, alpha_node_count)
    # Creat degree lists only private users.
    privae_degrees = np.delete(np.array(degrees), non_private).tolist()
    # Setting -1 of non private.
    flip_targets[non_private, ::] = -1
    flip_targets[::, non_private] = -1
    epsilon_one = toc_epsilon / 10
    epsilon_two = (toc_epsilon * 9) / 10
    laplace_scale_one = 1 / epsilon_one
    laplace_scale_two = 1 / epsilon_two
    result_list = []
    # Get indices of upper triangular matrix of private users.
    result_index_list = [
        n * i + j for i in range(n) for j in range(n)
        if j > i and flip_targets[i][j] != -1]
    # Get values of upper triangular matrix of private users.
    upper_aij = [aij[i][j] for i in range(n) for j in range(
        n) if j > i and flip_targets[i][j] != -1]
    non_private_edge_index = [
        n * i + j for i in range(n) for j in range(n)
        if flip_targets[i][j] == -1 and aij[i][j] == 1]
    # Compute d*
    np.random.seed()
    degree_laplace = np.random.laplace(
        loc=0.0, scale=laplace_scale_one, size=len(privae_degrees))
    lap_degrees = privae_degrees + degree_laplace
    np.random.seed()
    aij_laplace = np.random.laplace(
        loc=0.0, scale=laplace_scale_two, size=len(upper_aij))
    # Compute aij*
    result_list = upper_aij + aij_laplace
    t = math.floor((np.sum(lap_degrees)) / 2.0)
    if t < 0:
        t = 0
    elif t > len(result_index_list):
        t = len(result_index_list)
    t = int(t)
    sort_result_list = np.argsort(result_list)[::-1][0:t]
    target_index = [result_index_list[i] for i in sort_result_list]
    target_edge_mat = list(
        map(functools.partial(make_edgemat, n=n), target_index))
    non_private_edge_mat = list(map(functools.partial(
        make_non_private_edgemat, n=n), non_private_edge_index))
    target_edge_mat = list(itertools.chain.from_iterable(target_edge_mat))
    target_edge_mat.extend(non_private_edge_mat)
    use_neighbor_array = np.array(target_edge_mat)
    if t != 0:
        neighbors = list(map(functools.partial(
            get_neighbor, target_edge_mat=use_neighbor_array), node_index))
        max_degree = max([len(res) for res in neighbors])
        graph.max_neighbor = max_degree
        graph.neighbors = neighbors
        edge_mat = torch.LongTensor(target_edge_mat).transpose(0, 1)
    if t == 0:
        graph.max_neighbor = 0
        graph.neighbors = [[] for i in range(len(graph.g))]
        edge_mat = torch.LongTensor([[], []])
    graph.edge_mat = edge_mat
    # Exist only degree_as_tag=0
    if degree_as_tag == 1:
        print("")

    return graph


def make_edgemat(num, n):
    x_index = math.floor(num/n)
    y_index = num % n
    result = []
    if x_index != y_index:
        result = [[x_index, y_index], [y_index, x_index]]
    else:
        result = [[x_index, y_index]]
    return result


def get_neighbors(line):
    result = [i for i, x in enumerate(line) if x == 1]
    return result


def get_neighbor(num, target_edge_mat):
    target_array = target_edge_mat[target_edge_mat[:, 0] == num]
    result_list = sorted([res[1] for res in target_array])
    return result_list


def make_non_private_edgemat(num, n):
    x_index = math.floor(num/n)
    y_index = num % n
    result = [x_index, y_index]
    return result
