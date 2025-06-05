
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import MMCIFParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1


def three_to_one(resname):
    return protein_letters_3to1.get(resname.capitalize(), resname)


def get_sidechain_centroid(residue):
    side_chain_atoms = [
        atom for atom in residue if atom.get_name() not in ('N', 'CA', 'C', 'O')]
    if not side_chain_atoms:
        return None  
    coords = np.array([atom.get_coord() for atom in side_chain_atoms])
    return coords.mean(axis=0)

def get_alpha_carbon(residue):
    return residue['CA'].get_coord() if 'CA' in residue else None


def find_neighbors_all_carbons(target_residue, chain, radius):
    carbon_atoms = [atom for atom in target_residue if atom.element == 'C']
    if not carbon_atoms:
        return []

    neighbors = set()

    for carbon in carbon_atoms:
        carbon_coord = carbon.get_coord()
        for res in chain:
            if not is_aa(res) or res == target_residue:
                continue
            for atom in res:
                try:
                    dist = np.linalg.norm(carbon_coord - atom.get_coord())
                except Exception:
                    continue
                if dist <= radius:
                    resname = three_to_one(res.get_resname().strip())
                    neighbors.add((resname, res.id[1], dist))
                    break
    return list(neighbors)


def calculate_neighbors(structure_path, target_pos, chain_id='A', radius=10.0, method="centroid"):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", structure_path)
    model = structure[0]

    if chain_id not in model:
        return []

    chain = model[chain_id]
    target_res = next((res for res in chain if is_aa(res) and res.id[1] == target_pos), None)
    if not target_res:
        return []

    if method == "centroid":
        target_center = get_sidechain_centroid(target_res)
        if target_center is None:
            return []
        get_coord = get_sidechain_centroid
    elif method == "CA":
        target_center = get_alpha_carbon(target_res)
        if target_center is None:
            return []
        get_coord = get_alpha_carbon
    elif method == "carbon_atoms":
        return find_neighbors_all_carbons(target_res, chain, radius)
    else:
        raise ValueError(f"Unknown method: {method}")

    neighbors = []
    for res in chain:
        if not is_aa(res):
            continue
        pos = res.id[1]
        center = get_coord(res)
        if center is None:
            continue
        dist = np.linalg.norm(target_center - center)
        if dist <= radius:
            resname = three_to_one(res.get_resname().strip())
            neighbors.append((resname, pos, dist))

    return neighbors


def process_all_sites(mutation_file, structure_dir, output_csv, radius=10.0, method="centroid"):
    df = pd.read_csv(mutation_file, sep='\t')
    seen = set()
    tqdm.write(str(df.head()))
    with open(output_csv, "w") as f:
        f.write("transcript_id,ub_position,neighbor_residue,neighbor_position,distance\n")

    tqdm.write(f"Starting 3D neighbor search for {len(df)} Ub entries (radius = {radius} Å) using method '{method}'")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing ({method})"):
        transcript_id = row.get("transcript_id")
        ub_pos = row.get("Protein_position")
        uniprot_id = row.get("uniprot_ac")

        if pd.isna(transcript_id) or pd.isna(uniprot_id) or pd.isna(ub_pos):
            continue

        try:
            ub_pos = int(float(ub_pos))
        except:
            tqdm.write(f"Invalid position for {transcript_id} ({ub_pos}), skipping")
            continue

        key = (transcript_id, ub_pos, method)
        if key in seen:
            continue
        seen.add(key)

        structure_file = os.path.join(structure_dir, f"{uniprot_id}.cif")
        if not os.path.exists(structure_file):
        # Try prefix match (e.g., Q6PJG6_*.cif)
            try:
                matches = [f for f in os.listdir(structure_dir)
                            if f.startswith(f"{uniprot_id}_") and f.endswith(".cif")]
                if matches:
                    structure_file = os.path.join(structure_dir, matches[0])
                else:
                    tqdm.write(f"Structure not found for UniProt ID: {uniprot_id}")
                    continue
            except Exception as e:
                tqdm.write(f"Error accessing structure directory: {e}")
                continue
                if not os.path.exists(structure_file):
                    tqdm.write(f"Structure not found: {structure_file}")
                    continue

        try:
            neighbors = calculate_neighbors(structure_file, ub_pos, radius=radius, method=method)
        except Exception as e:
            tqdm.write(f"Error calculating neighbors for {uniprot_id} at {ub_pos}: {e}")
            continue

        if not neighbors:
            tqdm.write(f"No neighbors within {radius} Å for {transcript_id} at position {ub_pos}")
            continue

        with open(output_csv, "a") as f:
            for resname, pos, dist in neighbors:
                f.write(f"{transcript_id},{ub_pos},{resname},{pos},{dist:.3f}\n")
            f.flush()


def generate_all_neighbor_datasets(mutation_file, structure_dir, radius):

    process_all_sites(
        mutation_file=mutation_file,
        structure_dir=structure_dir,
        output_csv=f"ub_neighbors_CA_{radius}ang_pdb.csv",
        radius=radius,
        method="CA")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation_file", required=True, help="Path to ub_mut.tsv")
    parser.add_argument("--structure_dir", required=True, help="Directory with AlphaFold .cif files")
    parser.add_argument("--radius", type=float, default=10.0, help="Radius in Å for neighbor detection")
    args = parser.parse_args()

    generate_all_neighbor_datasets(
        mutation_file=args.mutation_file,
        structure_dir=args.structure_dir,
        radius=args.radius
    )
#python calc_3d_radius.py --mutation_file "../../../tcga_data/mut_at_ub_filtered.tsv" --structure_dir ../../../tcga_data/structures --radius 6
#python calc_3d_radius.py --mutation_file ../../../OneDrive - Oregon Health & Science University/Nguyen_2025_dissertation/Aim1/tcga_data/mut_at_ub_filtered.tsv --structure_dir ../../../OneDrive - Oregon Health & Science University/Nguyen_2025_dissertation/Aim1/tcga_data/pdb_struct --radius 6

