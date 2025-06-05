import os
import time
import torch
import logging
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score, matthews_corrcoef
from utils import get_args, create_logname, create_modelname, save_model, write_results_to_csv, custom_collate
from models import GCN, FocalLoss, GIN, GAT, GATEdgeAttr, GCNEdgeWt, GINEdgeWt
from data import ProteinDataset
from torch_geometric.data import Data
import pandas as pd
import csv


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

def compute_metrics(all_labels, all_probs):
    predictions = (np.array(all_probs) > 0.5).astype(float)
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
        specificity = tn / (tn + fp)
    except ValueError:
        specificity = 0.0

    return {
        "accuracy": accuracy_score(all_labels, predictions),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "f1": f1_score(all_labels, predictions),
        "precision": precision_score(all_labels, predictions),
        "recall": recall_score(all_labels, predictions),
        "aupr": average_precision_score(all_labels, all_probs),
        "mcc": matthews_corrcoef(all_labels, predictions),
        "specificity": specificity,
    }

def train(trainloader, model, optimizer, crit, device='cuda'):
    model.train()
    total_loss = 0
    requires_edge_attr = isinstance(model, (GCNEdgeWt, GINEdgeWt, GATEdgeAttr))
    all_labels, all_probs = [], []

    for data in trainloader:
        optimizer.zero_grad()
        data = data.to(device)
        if requires_edge_attr:
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
        for data in dataloader:
            data = data.to(device)
            if requires_edge_attr:
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

def predict(model, folder, output_csv, device='cuda'):
    model.eval()
    requires_edge_attr = isinstance(model, (GCNEdgeWt, GATEdgeAttr, GINEdgeWt))

    files = sorted([f for f in os.listdir(folder) if f.endswith('.pt') and f.startswith('secvec_')])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    predicted_proteins = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'protein' in row:
                    predicted_proteins.add(row['protein'])

    mode = 'a' if os.path.exists(output_csv) else 'w'
    with open(output_csv, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['protein', 'position', 'probability'])
        if mode == 'w':
            writer.writeheader()

        for fname in files:
            if fname in predicted_proteins:
                continue

            file_path = os.path.join(folder, fname)
            try:
                data = torch.load(file_path)
                if data.x.shape[0] == 3 and data.x.shape[2] == 1024:
                    data.x = data.x[1]
                if hasattr(data, 'ub_label'):
                    data.y = torch.tensor(data.ub_label, dtype=torch.float)
                else:
                    data.y = torch.zeros(data.x.shape[0], dtype=torch.float)
                if not hasattr(data, 'lys_indic'):
                    continue

                data = data.to(device)
                if requires_edge_attr:
                    out = model(data.x, data.edge_index, data.edge_attr).squeeze()
                else:
                    out = model(data.x, data.edge_index).squeeze()

                lys_mask = (data.lys_indic == 1)
                if lys_mask.sum().item() == 0:
                    continue

                probs = torch.sigmoid(out[lys_mask])
                lys_indices = torch.where(lys_mask)[0].tolist()
                prob_values = probs.detach().cpu().tolist()

                for pos, prob in zip(lys_indices, prob_values):
                    writer.writerow({'protein': fname, 'position': pos, 'probability': prob})
            except Exception as e:
                logging.warning(f"Skipping {fname} due to error: {str(e)}")

if __name__ == '__main__':
    time0 = time.time()
    args = get_args()
    base_path = "/home/exacloud/gscratch/WuLab/nguyjust"
    log_filename = create_logname(base_path, args.model, args.lr, args.wd, args.hidden, args.dropout)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_filename, filemode="a")

    device = 'cuda' if torch.cuda.is_available() and not args.ignore_cuda else 'cpu'
    model = select_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = FocalLoss(alpha=0.8, gamma=2.5)

    folder_path="/home/exacloud/gscratch/WuLab/nguyjust/process_edge_new/secvec_sort/"
    train_files = [(f, 'train') for f in os.listdir(os.path.join(folder_path, "train")) if not f.startswith('.')]
    val_files = [(f, 'val') for f in os.listdir(os.path.join(folder_path, "val")) if not f.startswith('.')]
    test_files = [(f, 'test') for f in os.listdir(os.path.join(folder_path, "test")) if not f.startswith('.')]

    train_dataset = ProteinDataset(ids=train_files, folder_path=folder_path, shuffle_edges=args.shuffle_edges)
    val_dataset = ProteinDataset(ids=val_files, folder_path=folder_path, shuffle_edges=args.shuffle_edges)
    test_dataset = ProteinDataset(ids=test_files, folder_path=folder_path, shuffle_edges=args.shuffle_edges)

    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch, num_workers=args.workers,
                            shuffle=True, collate_fn=custom_collate)
    valloader = DataLoader(dataset=val_dataset, batch_size=args.batch, num_workers=args.workers,
                            shuffle=False, collate_fn=custom_collate)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch, num_workers=args.workers,
                            shuffle=False, collate_fn=custom_collate)

    #best_f1 = 0.0
    best_auc = 0.0
    best_epoch = -1
    models_dir = os.path.join(base_path, "sv_models")
    os.makedirs(models_dir, exist_ok=True)
    run_id = f"{args.model}_lr{args.lr}_wd{args.wd}_hid{args.hidden}_drop{args.dropout}"
    #best_model_path = os.path.join(models_dir, f"best_{run_id}_f1.pth")
    best_model_path = os.path.join(models_dir, f"best_{run_id}_auc.pth")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train(trainloader, model, optimizer, crit, device=device)
        val_loss, val_metrics = evaluate(model, valloader, crit, device=device)

        logging.info(
            f"Epoch {epoch}/{args.epochs}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train AUC: {train_metrics['roc_auc']:.4f}, Val AUC: {val_metrics['roc_auc']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
            f"Train Precision: {train_metrics['precision']:.4f}, Val Precision: {val_metrics['precision']:.4f}, "
            f"Train Recall: {train_metrics['recall']:.4f}, Val Recall: {val_metrics['recall']:.4f}, "
            f"Train MCC: {train_metrics['mcc']:.4f}, Val MCC: {val_metrics['mcc']:.4f}, "
            f"Train AUPR: {train_metrics['aupr']:.4f}, Val AUPR: {val_metrics['aupr']:.4f}, "
            f"Train Specificity: {train_metrics['specificity']:.4f}, Val Specificity: {val_metrics['specificity']:.4f}"
        )

        if val_metrics['roc_auc'] > best_auc:
        #if val_metrics['f1'] > best_f1:
            #best_f1 = val_metrics['f1']

            best_auc = val_metrics['roc_auc']
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            #logging.info(f"New best model saved with F1: {best_f1:.4f} at epoch {epoch}")
            logging.info(f"New best model saved with AUC: {best_auc:.4f} at epoch {epoch}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    logging.info(f"Running test on best model from epoch {best_epoch}")
    test_loss, test_metrics = evaluate(model, testloader, crit, device=device)
    logging.info(
        f"FINAL TEST EVAL — Test Loss: {test_loss:.4f}, "
        f"Test Acc: {test_metrics['accuracy']:.4f}, Test AUC: {test_metrics['roc_auc']:.4f}, "
        f"Test F1: {test_metrics['f1']:.4f}, Test Precision: {test_metrics['precision']:.4f}, "
        f"Test Recall: {test_metrics['recall']:.4f}, Test MCC: {test_metrics['mcc']:.4f}, "
        f"Test AUPR: {test_metrics['aupr']:.4f}, Test Specificity: {test_metrics['specificity']:.4f}"
    )
    predict_folder = "/home/exacloud/gscratch/WuLab/nguyjust/process_edge_human/secvec"
    output_csv = f"{base_path}/predictions/human_preds_{args.model}.csv"
    #predict(model, predict_folder, output_csv, device=device)
    
    t_time = (time.time()-time0)/3600
    logging.info(f'Total script time: {t_time:.3f} hrs')
    logging.info('End training script')

