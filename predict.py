import os
import csv
import torch
import numpy as np
import logging
import traceback
from models import GAT

def load_model(model_class, model_path, device, **kwargs):
    print(">>> Loading model")
    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(">>> Model loaded")
    return model


def predict(model, folder, output_csv, device='cuda', threshold=0.39):
    print(f">>> Entering predict() with folder: {folder}")
    assert os.path.exists(folder), f"Input folder does not exist: {folder}"

    model.eval()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    files = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])
    print(f">>> Found {len(files)} .pt files")

    seen = set()
    if os.path.exists(output_csv):
        print(f">>> Reading existing CSV: {output_csv}")
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    seen.add((row['protein'], int(row['position'])))
                except (KeyError, ValueError):
                    continue
        print(f">>> Found {len(seen)} existing predictions")

    mode = 'a' if os.path.exists(output_csv) else 'w'
    with open(output_csv, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['protein', 'position', 'probability'])
        if mode == 'w':
            writer.writeheader()

        for fname in files:
            print(f"Processing {fname}")
            protein_id = fname.replace('.pt', '').replace('secvec_', '')
            path = os.path.join(folder, fname)
            try:
                data = torch.load(path)

                if isinstance(data.x, np.ndarray):
                    print("  - Converting x from numpy to torch tensor")
                    data.x = torch.tensor(data.x, dtype=torch.float32)
                elif isinstance(data.x, tuple):
                    print("  - Fixing x: tuple detected")
                    data.x = data.x[0]
                    if isinstance(data.x, np.ndarray):
                        data.x = torch.tensor(data.x, dtype=torch.float32)

                if isinstance(data.x, torch.Tensor) and data.x.ndim == 3 and data.x.shape[0] == 3:
                    print(f"  - Collapsing 3D tensor x from shape {data.x.shape}")
                    data.x = data.x[1]

                print(f"  - Final x type: {type(data.x)}, shape: {data.x.shape}, dtype: {data.x.dtype}")
                print(f"  - edge_index shape: {data.edge_index.shape}")
                print(f"  - edge_attr shape: {data.edge_attr.shape if hasattr(data, 'edge_attr') else 'None'}")

                data = data.to(device)
                out = model(data.x, data.edge_index).squeeze()
                print(f"  - Model forward passed, output shape: {out.shape}")

                if not hasattr(data, 'lys_indic') or data.lys_indic.sum().item() == 0:
                    print("  - Skipping: no lysine sites")
                    continue

                lys_mask = data.lys_indic == 1


                probs = torch.sigmoid(out[lys_mask])
                lys_positions = torch.where(lys_mask)[0].tolist()
                prob_values = probs.detach().cpu().tolist()

                for pos, prob in zip(lys_positions, prob_values):
                    if prob >= threshold:
                        key = (protein_id, pos)
                        if key not in seen:
                            writer.writerow({'protein': protein_id, 'position': pos, 'probability': prob})
                            seen.add(key)

            except Exception as e:
                msg = f"Skipping {fname} due to error: {str(e)}"
                logging.warning(msg)
                print(msg)
                traceback.print_exc()

if __name__ == "__main__":
    print(">>> START OF SCRIPT")
    print(">>> Running from:", os.path.abspath(__file__))

    log_dir = "/home/exacloud/gscratch/WuLab/nguyjust/predictions"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(log_dir, "predict_gat.log"),
        filemode='a'
    )

    model_path = "/home/exacloud/gscratch/WuLab/nguyjust/sv_models/best_gat_lr0.0005_wd0.0_hid256_drop0.1_f1.pth"
    input_folder = "/home/exacloud/gscratch/WuLab/nguyjust/process_edge_human/secvec"
    output_csv = "/home/exacloud/gscratch/WuLab/nguyjust/predictions/human_preds_gat_0.39.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(GAT, model_path, device, in_channels=1024, hidden_channels=256, dropout_rate=0.1)
    predict(model, input_folder, output_csv, device=device)

