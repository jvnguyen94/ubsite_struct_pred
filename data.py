import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import custom_collate


class ProteinDataset(Dataset):
    def __init__(self, ids, folder_path, shuffle_edges=False):
        super().__init__()
        self.ids = ids
        self.folder_path = folder_path
        self.shuffle_edges = shuffle_edges

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname, origin = self.ids[idx]
        file_path = os.path.join(self.folder_path, origin, fname)
        data = torch.load(file_path)
        # Ensure x is in correct shape: extract only the second element from (3, L, 1024) -> (L, 1024)
        if data.x.shape[0] == 3 and data.x.shape[2] == 1024:
            data.x = data.x[1]
        if hasattr(data, 'ub_label'):
            data.y = torch.tensor(data.ub_label, dtype=torch.float)
        else:
            # Assign default label
            data.y = torch.zeros(data.x.shape[0], dtype=torch.float)

        edge_index = data.edge_index
        if self.shuffle_edges:
            # num_edges = data.edge_index.shape[1]  # Number of edges
            # perm = torch.randperm(num_edges)  # Create a random permutation
            # data.edge_index = data.edge_index[:, perm]  # Apply permutation
            data['edge_index'][0] = data['edge_index'][0][torch.randperm(
                data['edge_index'].shape[1])]  # Shuffle src
            data['edge_index'][1] = data['edge_index'][1][torch.randperm(
                data['edge_index'].shape[1])]  # Shuffle dest
            not_dupe = edge_index[0] != edge_index[1]  # where src != dst
            edge_index = edge_index[:, not_dupe]
        num_lys = int(data.lys_indic.sum().item())
        num_pos = int((data.y[data.lys_indic == 1] == 1).sum().item())
        num_neg = num_lys - num_pos
        data.edge_index = edge_index
        #print(f"Loaded {fname}: Lysines: {num_lys}, Ubiquitinated (1): {num_pos}, Non-Ubiquitinated (0): {num_neg}")

        return data
