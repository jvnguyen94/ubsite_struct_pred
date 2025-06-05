import os
import time
import torch
import logging
import warnings
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score, matthews_corrcoef
)
from utils import (
    get_args, create_logname, create_modelname, save_model,
    write_results_to_csv, custom_collate
)
from models import GCN, FocalLoss, GIN, GAT, GATEdgeAttr, GCNEdgeWt, GINEdgeWt
from data import ProteinDataset

def select_model(args):
    if args.model == 'gcn':
        return GCN(in_channels=1024, hidden_channels=args.hidden, dropout_rate=args.dropout)
    elif args.model == 'gin':
        return GIN(in_channels=1024, hidden_channels=args.hidden, dropout_rate=args.dropout)
    elif args.model == 'gat':
        return GAT(in_channels=1024, hidden_channels=args.hidden, dropout_rate=args.dropout)
    elif args.model == 'gcnedge':
        return GCNEdgeWt(in_channels=1024, hidden_channels=args.hidden, dropout_rate=args.dropout)
    elif args.model == 'ginedge':
        return GATEdgeAttr(in_channels=1024, hidden_channels=args.hidden, dropout_rate=args.dropout)
    elif args.model == 'gatedge':
        return GINEdgeWt(in_channels=1024, hidden_channels=args.hidden, dropout_rate=args.dropout)

def train(trainloader, model, optimizer, crit, device='cuda'):
    model.train()
    total_loss = 0
    requires_edge_attr = isinstance(model, (GCNEdgeWt, GINEdgeWt, GATEdgeAttr))

    all_labels, all_probs = [], []

    for batch_idx, data in enumerate(trainloader):
        optimizer.zero_grad()
        data.x, data.edge_index, data.y, data.lys_indic = (
            data.x.to(device),
            data.edge_index.to(device),
            data.y.to(device),
            data.lys_indic.to(device),
        )

        if requires_edge_attr:
            data.edge_attr = data.edge_attr.to(device)
            out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        else:
            out = model(data.x, data.edge_index).squeeze()

        lys_mask = (data.lys_indic == 1)
        out_lys = out[lys_mask]
        y_lys = data.y[lys_mask]
        if y_lys.numel() > 0:
            loss = crit(out_lys, y_lys.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            probs = torch.sigmoid(out_lys)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(y_lys.cpu().numpy())

    return total_loss / len(trainloader), compute_metrics(all_labels, all_probs)

def evaluate(model, dataloader, criterion, device='cuda'):
    model.eval()
    total_loss = 0
    all_labels, all_probs = [], []
    requires_edge_attr = isinstance(model, (GCNEdgeWt, GATEdgeAttr, GINEdgeWt))

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data.x, data.edge_index, data.y, data.lys_indic = (
                data.x.to(device),
                data.edge_index.to(device),
                data.y.to(device),
                data.lys_indic.to(device),
            )
            if requires_edge_attr:
                data.edge_attr = data.edge_attr.to(device)
                yhat = model(data.x, data.edge_index, data.edge_attr).squeeze()
            else:
                yhat = model(data.x, data.edge_index).squeeze()

            lys_mask = (data.lys_indic == 1)
            yhat_lys = yhat[lys_mask]
            y_lys = data.y[lys_mask]
            if y_lys.numel() > 0:
                loss = criterion(yhat_lys, y_lys.float())
                total_loss += loss.item()
                probs = torch.sigmoid(yhat_lys)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_lys.cpu().numpy())

    return total_loss / len(dataloader), compute_metrics(all_labels, all_probs)

def compute_metrics(all_labels, all_probs):
    thresholds = [0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results_by_threshold = {}

    for threshold in thresholds:
        predictions = (np.array(all_probs) > threshold).astype(float)
        try:
            tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
            specificity = tn / (tn + fp)
        except ValueError:
            specificity = 0.0

        results_by_threshold[threshold] = {
            "accuracy": accuracy_score(all_labels, predictions),
            "roc_auc": roc_auc_score(all_labels, all_probs),
            "f1": f1_score(all_labels, predictions),
            "precision": precision_score(all_labels, predictions),
            "specificity": specificity,
            "recall": recall_score(all_labels, predictions),
            "aupr": average_precision_score(all_labels, all_probs),
            "mcc": matthews_corrcoef(all_labels, predictions),
        }

    return results_by_threshold

if __name__ == '__main__':
    time0 = time.time()
    args = get_args()
    base_path = "/home/exacloud/gscratch/WuLab/nguyjust"
    log_filename = create_logname(base_path, args.model, args.lr, args.wd, args.hidden, args.dropout)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_filename, filemode="a")
    logging.info(f'hidden layers: {args.hidden}, lr: {args.lr}, wd: {args.wd}, dropout: {args.dropout}')

    device = 'cuda' if torch.cuda.is_available() and not args.ignore_cuda else 'cpu'
    model = select_model(args).to(device)
    logging.info(f"Using model: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = FocalLoss(alpha=0.8, gamma=2.5)
    logging.info("Model, optimizer, and evaluation setup complete")

    folder_path = "/home/exacloud/gscratch/WuLab/nguyjust/process_edge_new/secvec_sort/"
    train_files = [(f, 'train') for f in os.listdir(os.path.join(folder_path, "train")) if not f.startswith('.')]
    val_files = [(f, 'val') for f in os.listdir(os.path.join(folder_path, "val")) if not f.startswith('.')]

    train_dataset = ProteinDataset(ids=train_files, folder_path=folder_path, shuffle_edges=args.shuffle_edges)
    val_dataset = ProteinDataset(ids=val_files, folder_path=folder_path, shuffle_edges=args.shuffle_edges)

    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch,
                            num_workers=args.workers, shuffle=True, collate_fn=custom_collate)
    valloader = DataLoader(dataset=val_dataset, batch_size=args.batch,
                            num_workers=args.workers, shuffle=False, collate_fn=custom_collate)
    
    test_files = [(f, 'test') for f in os.listdir(os.path.join(folder_path, "test")) if not f.startswith('.')]
    test_dataset = ProteinDataset(ids=test_files, folder_path=folder_path, shuffle_edges=args.shuffle_edges)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch,
                        num_workers=args.workers, shuffle=False, collate_fn=custom_collate)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train(trainloader, model, optimizer, crit, device=device)
        val_loss, val_metrics = evaluate(model, valloader, crit, device=device)

        for threshold in train_metrics:
            tm = train_metrics[threshold]
            vm = val_metrics[threshold]
            logging.info(
                f"Epoch {epoch}/{args.epochs}, Threshold: {threshold:.1f}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train AUC: {tm['roc_auc']:.4f}, Val AUC: {vm['roc_auc']:.4f}, "
                f"Train Acc: {tm['accuracy']:.4f}, Val Acc: {vm['accuracy']:.4f}, "
                f"Train F1: {tm['f1']:.4f}, Val F1: {vm['f1']:.4f}, "
                f"Train Precision: {tm['precision']:.4f}, Val Precision: {vm['precision']:.4f}, "
                f"Train Recall: {tm['recall']:.4f}, Val Recall: {vm['recall']:.4f}, "
                f"Train MCC: {tm['mcc']:.4f}, Val MCC: {vm['mcc']:.4f}, "
                f"Train AUPR: {tm['aupr']:.4f}, Val AUPR: {vm['aupr']:.4f}, "
                f"Train Specificity: {tm['specificity']:.4f}, Val Specificity: {vm['specificity']:.4f}"
            )
    test_loss, test_metrics = evaluate(model, testloader, crit, device=device)
    for threshold in test_metrics:
        te = test_metrics[threshold]
        logging.info(
            f"FINAL TEST EVAL — Threshold: {threshold:.1f}, Test Loss: {test_loss:.4f}, "
            f"Test Acc: {te['accuracy']:.4f}, Test AUC: {te['roc_auc']:.4f}, Test F1: {te['f1']:.4f}, "
            f"Test Precision: {te['precision']:.4f}, Test Recall: {te['recall']:.4f}, "
            f"Test MCC: {te['mcc']:.4f}, Test AUPR: {te['aupr']:.4f}, Test Specificity: {te['specificity']:.4f}"
        )
    t_time = (time.time()-time0)/60/60
    logging.info(f'Total script time: {t_time:.3f} hrs')
    logging.info('End training script')
