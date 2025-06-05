##
import torch_geometric.nn as pyg_nn
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, dropout_rate=0.3):
        super(GIN, self).__init__()
        self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(
            in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
        self.conv2 = pyg_nn.GINConv(nn.Sequential(nn.Linear(
            hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
        self.conv3 = pyg_nn.GINConv(
            nn.Sequential(nn.Linear(hidden_channels, 1)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x
    

class GINEdgeWt(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, dropout_rate=0.3):
        super(GINEdgeWt, self).__init__()

        # GINConv layers with learnable MLPs
        self.conv1 = pyg_nn.GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ), train_eps=True  # Ensures learnable aggregation
        )
        self.conv2 = pyg_nn.GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ), train_eps=True
        )
        self.conv3 = pyg_nn.GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, 1)
            ), train_eps=True
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.shape[1],), device=x.device)  # Correct shape
        else:
            edge_weight = edge_weight.view(-1)  # Ensures it's 1D: (num_edges,)

        x = self.conv1(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)

        return x
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, dropout_rate=0.2):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = pyg_nn.GCNConv(hidden_channels, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x


class GCNEdgeWt(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, dropout_rate=0.2):
        super(GCNEdgeWt, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = pyg_nn.GCNConv(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.shape[1], 1), device=x.device)

        x = self.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=8, dropout_rate=0.4, heads=4):
        super(GAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = pyg_nn.GATConv(
            hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = pyg_nn.GATConv(hidden_channels * heads, 1, heads=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x
    

class GATEdgeAttr(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=8, dropout_rate=0.4, heads=4, edge_dim=1):
        super(GATEdgeAttr, self).__init__()
        self.conv1 = pyg_nn.GATConv(
            in_channels, hidden_channels, heads=heads, edge_dim=1)
        self.conv2 = pyg_nn.GATConv(
            hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1)
        self.conv3 = pyg_nn.GATConv(hidden_channels * heads, 1, heads=1, edge_dim=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # Ensure edge_attr is provided, or default to ones
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.shape[1], 1), device=x.device)

        x = self.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        return x


class GCN_edge_prev(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate):
        super(GCN_edge_prev, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device)
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        return x
