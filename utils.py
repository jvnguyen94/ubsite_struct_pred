import os
import csv
import logging
import argparse
from datetime import datetime
import torch
from torch_geometric.data import Batch


def get_args():
    parser = argparse.ArgumentParser(
        description="Train GCN with hyperparameters")
    parser.add_argument("--batch", type=int, default=50,
                        help="training batch size")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers for dataloaders")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units')
    parser.add_argument('--dropout', type=float,
                        default=0., help='Dropout rate')
    parser.add_argument('--shuffle_edges', action="store_true",
                        help="Shuffle graph edges during training")
    parser.add_argument('--model', type=str, choices=['gcn', 'gin', 'gat', 'gcnedge', 'gatedge', 'ginedge'], 
                        default='gcn', help="Model selection")


    return parser.parse_args()


def create_logname(base_path, model, lr, wd, hidden, dropout):
    date_str = datetime.now().strftime("%Y%m%d")
    filename=f"{base_path}/output/{date_str}_{model}_lr{lr}_wd{wd}_hidden{hidden}_dropout{dropout}_lysonly_sv.log"
    return filename


def create_modelname(base_path, lr, wd, hidden, dropout):
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{base_path}/models/{date_str}_lr{lr}_wd{wd}_hidden{hidden}_dropout{dropout}_.pth"
    return filename


def save_model(model, path):
    torch.save(model.state_dict(), path)


def write_results_to_csv(results, filename):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def custom_collate(batch):
    return Batch.from_data_list(batch)
