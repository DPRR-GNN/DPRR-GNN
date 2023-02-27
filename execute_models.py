import argparse
import copy
import itertools
import math
import os
import random
from pathlib import Path

from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from more_itertools import chunked
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
# from torch_geometric.seed import seed_everything
from insert_noise import insert_noise_use_C_use_map
from models.graphcnn import GraphCNN
from utils import (EarlyStopping, HoldPerformanceIndices, get_params,
                   get_y_data, load_data, output_board)

criterion = nn.CrossEntropyLoss()

output_headers = ['dataset', 'case', 'seed', 'alpha', 'epsilon',
                  'batch_size', 'adam_lr', 'num_layers', 'dropout',
                  'hidden_unit', 'model_epoch', 'validation_acc',
                  'traindata_loss_training', 'evaldata_loss_training',
                  'accuracy_train', 'accuracy_test',
                  'train_auc', 'test_auc']
#log_dir = Path('./logs')
output_dir = Path('./output')


def train(model, device, train_graphs, optimizer, batch_s):
    model.train()
    '''
    Training model and compute loss.
    Args:
        model(nn.Module): GNN model
        device(str): Device for calculation(cuda or cpu).
        train_graphs(list(S2VGraph)): Graph classes.
        optimizer(optim): Optimizer for training.
        batch_s(int): Batch size.
    '''
    graph_indexs = [i for i in range(len(train_graphs))]
    random.seed()
    random.shuffle(graph_indexs)
    # Create chank list each batch size.
    chank_list = list(chunked(graph_indexs, batch_s))
    total_lengths = math.floor(len(train_graphs) / batch_s)
    loss_accum = 0
    for i in range(total_lengths):
        selected_idx = chank_list[i]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)
        labels = torch.LongTensor(
            [graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    average_loss = loss_accum/total_lengths

    return average_loss


def test(model, device, train_graphs, test_graphs, val_graphs):
    '''
    Compute accuracy and roc with model.
    Args
        model(nn.Module): GNN model
        device(str): Device for calculation(cuda or cpu).
        train_graphs(list(S2VGraph)): Graph classes(train).
        test_graphs(list(S2VGraph)): Graph classes(test).
        val_graphs(list(S2VGraph)): Graph classes(valid).
    '''
    model.eval()

    def evaluate(graphs):
        with torch.no_grad():
            true_answer_list = [graph.label for graph in graphs]
            output = pass_data_iteratively(graphs)
            pred = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor(
                [graph.label for graph in graphs]).to(device)
            # output = torch.exp(output) / \
            #    torch.sum(torch.exp(output), 1, keepdim=True)
            output = F.softmax(output, 1)
            if output.size(1) == 2:
                output = output[:, 1]

            output = output.tolist()
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(graphs))
            roc = roc_auc_score(
                true_answer_list, output, multi_class='ovr')

        return acc, roc

    def pass_data_iteratively(graphs, minibatch_size=64):
        # model.eval()
        output = []
        idx = np.arange(len(graphs))
        for i in range(0, len(graphs), minibatch_size):
            sampled_idx = idx[i:i+minibatch_size]
            if len(sampled_idx) == 0:
                continue
            output.append(model([graphs[j] for j in sampled_idx]).detach())
        return torch.cat(output, 0)

    acc_train, train_roc = evaluate(train_graphs)
    acc_val, val_roc = evaluate(val_graphs)
    acc_test, test_roc = evaluate(test_graphs)
    '''
    print("accuracy train: %f test: %f val: %f rac train %f test: %f val: %f" %
          (acc_train, acc_test, acc_val, train_roc, test_roc, val_roc))
    '''

    return acc_train, acc_test, acc_val, train_roc, test_roc, val_roc


def val(model, device, val_graphs, batch_s):
    '''
    Compute loss.
    Args:
        model(nn.Module): GNN model
        device(str): Device for calculation(cuda or cpu).
        val_graphs(list(S2VGraph)): Graph classes.
        batch_s(int): Batch size.
    '''
    model.eval()
    graph_indexs = [i for i in range(len(val_graphs))]
    random.seed()
    random.shuffle(graph_indexs)
    # Create chank list each batch size.
    chank_list = list(chunked(graph_indexs, batch_s))
    total_lengths = math.ceil(len(val_graphs) / batch_s)
    with torch.no_grad():
        loss_accum = 0
        for i in range(total_lengths):
            selected_idx = chank_list[i]

            batch_graph = [val_graphs[idx] for idx in selected_idx]
            output = model(batch_graph)

            labels = torch.LongTensor(
                [graph.label for graph in batch_graph]).to(device)

            # compute loss
            loss = criterion(output, labels)

            loss = loss.detach().cpu().numpy()
            loss_accum += loss

        average_loss = loss_accum/total_lengths

    return average_loss


def main():
    parser = argparse.ArgumentParser(
        description='Conf file path')
    parser.add_argument('--conf_file', type=str, help='config file path')
    args = parser.parse_args()
    config = get_params(args.conf_file)
    dataset_name = config['dataset']
    assert config['train']['execution']['num_epoch'] \
        >= config['train']['execution']['minimum_stop_epoch']
    # epsilon = config['apply_noise'].get('epsilon', 0)
    output_filename = "GIN_powerful_gnn_" + \
        f"{config['apply_noise']['type']}_" + f"{dataset_name}.csv"
    param_iters = itertools.product(
        config['train']['execution']['seeds'],
        config['train']['execution']['alphas'], config['apply_noise']['epsilons'])
    degree_as_tag = 1 if config['model']['degree_as_tag'] else 0
    graphs, num_classes = load_data(
        dataset_name, config['model']['degree_as_tag'])
    print("dataset read end")
    dt_now = datetime.now()
    dt_now = dt_now.strftime('%Y%m%d%H%M%S')
    output_root = output_dir / \
        f'{config["apply_noise"]["type"]}_{dataset_name}_{dt_now}'
    output_csv_path = output_root / \
        f'gnn_{config["apply_noise"]["type"]}_{dataset_name}.csv'
    for idx, (seed, alpha, epsilon) in enumerate(param_iters):
        output_model_dir = output_root / \
            f'seed:{seed}_alpha:{alpha}_epsilon:{epsilon}'
        os.makedirs(output_model_dir, exist_ok=True)

        device = torch.device(
            f"cuda:{config['train']['execution']['device_num']}") \
            if torch.cuda.is_available() else torch.device("cpu")
        dataset = copy.deepcopy(graphs)
        if config['apply_noise']['type'] != 'no_noise':
            dataset = insert_noise_use_C_use_map(
                dataset, alpha, seed, config['apply_noise']['type'],
                degree_as_tag, epsilon=epsilon)
        # seed_everything(seed)
        random.shuffle(dataset)
        y_list = list(map(get_y_data, dataset))
        valid_ratio = config['train']['execution']['valid_ratio']
        test_ratio = config['train']['execution']['test_ratio']
        train_dataset, test_and_val_dataset = train_test_split(
            dataset, test_size=valid_ratio+test_ratio,
            stratify=y_list, random_state=seed)
        test_val_y_list = list(map(get_y_data, test_and_val_dataset))
        val_dataset, test_dataset = train_test_split(
            test_and_val_dataset,
            test_size=(test_ratio/(valid_ratio+test_ratio)),
            stratify=test_val_y_list, random_state=seed)
        print("make separate end")
        early_stopping = EarlyStopping(
            patience=config['train']['execution']['patience'], verbose=False)
        hold_performance_indices = HoldPerformanceIndices()
        model = GraphCNN(config['model']['num_layers'],
                         config['model']['num_mlp_layers'],
                         train_dataset[0].node_features.shape[1],
                         config['model']['hidden_unit'],
                         num_classes, config['model']['dropout'],
                         config['model']['learn_eps'],
                         config['model']['graph_pooling_type'],
                         config['model']['neighbor_pooling_type'],
                         device).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=config['train']['optimizer']['lr'])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config['train']['optimizer']['step_size'],
            gamma=config['train']['optimizer']['gamma'])
        for epoch in range(1, config['train']['execution']['num_epoch']+1):
            train_loss = train(model, device,
                               train_dataset, optimizer,
                               config['train']['execution']['batch_s'])
            val_loss = val(model, device, val_dataset,
                           config['val']['batch_s'])
            scheduler.step()
            train_acc, _, val_acc, _, _, _ = test(
                model, device, train_dataset, test_dataset, val_dataset)
            if epoch >= config['train']['execution']['minimum_stop_epoch']:
                early_result = early_stopping(val_loss, model)
                hold_performance_indices(epoch, train_acc, val_acc, train_loss,
                                         val_loss, early_result)
                if early_stopping.early_stop or \
                        epoch == config['train']['execution']['num_epoch']:
                    if early_stopping.early_stop:
                        print("early stooping")

                    _, best_test_acc, _, best_train_auc, best_test_auc, _ \
                        = test(early_stopping.model, device, train_dataset,
                               test_dataset, val_dataset)
                    final_result = [[dataset_name,
                                     config['apply_noise']['type'], seed,
                                     alpha, epsilon,
                                     config['train']['execution']['batch_s'],
                                     config['train']['optimizer']['lr'],
                                     config['model']['num_layers'],
                                     config['model']['dropout'],
                                     config['model']['hidden_unit'],
                                     hold_performance_indices.maxnum_epoch,
                                     hold_performance_indices.best_val_acc,
                                     hold_performance_indices.best_train_loss,
                                     hold_performance_indices.best_val_loss,
                                     hold_performance_indices.best_train_acc,
                                     best_test_acc, best_train_auc,
                                     best_test_auc
                                     ]]
                    df_final_result = pd.DataFrame(
                        final_result, columns=output_headers)
                    if idx == 0:
                        df_final_result.to_csv(output_csv_path, index=False)
                    else:
                        df_final_result.to_csv(
                            output_csv_path, mode='a', header=False,
                            index=False)

                    # os.makedirs(log_dir, exist_ok=True)
                    log_dir = output_model_dir / 'logs'
                    output_board(hold_performance_indices,
                                 log_dir)
                    ouput_model_path = output_model_dir / 'model_weight.pth'
                    torch.save(early_stopping.model.state_dict(),
                               ouput_model_path)
                    break

            else:
                hold_performance_indices(epoch, train_acc, val_acc, train_loss,
                                         val_loss, False)

            print(
                f'Epoch: {epoch:03d}, val_Loss: {val_loss:.4f}, ',
                f'train_Loss: {train_loss:.4f}, ' f'Val Acc: {val_acc:.4f}')


if __name__ == '__main__':
    main()
