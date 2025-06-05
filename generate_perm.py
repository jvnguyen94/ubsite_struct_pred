#%%
import os
import numpy as np
import pandas as pd
import time
import sys
import argparse
#%%
def process_transcript(transcript_id, cancer_type_filter, muts, output_dir, n_permutations, verbose, i, total):
    file_name = f"{transcript_id}_{cancer_type_filter}.npy"
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path):
        if verbose:
            print(f"[{i}/{total}] Skipping {file_name} (already exists)", flush=True)
        return

    if muts.empty:
        return

    try:
        protein_length = int(muts["protein_length"].iloc[0])
        observed_positions = muts["Protein_position"].astype(int).values
    except:
        return

    patient_counts = muts.groupby("patient_barcode").size().to_dict()
    total_mutations = sum(patient_counts.values())

    mutation_counts_permutations = np.random.randint(
        1, protein_length + 1,
        size=(n_permutations * total_mutations),
        dtype=np.int32
    )

    position_counts = np.bincount(
        mutation_counts_permutations,
        minlength=protein_length + 1
    ).astype(np.uint32)

    np.save(file_path, position_counts)

    if verbose:
        print(f"[{i}/{total}] {transcript_id} ({cancer_type_filter}) | Saved: {file_name}", flush=True)

def generate_permutation_counts(
    mutations_df,
    n_permutations=1000000,
    output_dir="perm_results",
    by_cancer=True,
    verbose=True
):
    os.makedirs(output_dir, exist_ok=True)

    transcripts = mutations_df["transcript_id"].unique()
    loop_iter = []

    if by_cancer:
        for transcript_id in transcripts:
            cancers = mutations_df[mutations_df["transcript_id"] == transcript_id]["cancer type"].unique()
            for cancer in cancers:
                loop_iter.append((transcript_id, cancer))
    else:
        loop_iter = [(tid, None) for tid in transcripts]

    total = len(loop_iter)
    start_time = time.time()

    for i, (transcript_id, cancer_type_filter) in enumerate(loop_iter, 1):
        iter_start = time.time()

        muts = mutations_df[mutations_df["transcript_id"] == transcript_id]
        if by_cancer:
            muts = muts[muts["cancer type"] == cancer_type_filter]

        process_transcript(transcript_id, cancer_type_filter, muts, output_dir, n_permutations, verbose, i, total)

        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (total - i)
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
        print(f"[ETA] Completed {i}/{total} | Elapsed: {elapsed:.2f}s | ETA: {eta}", flush=True)

    print("\nPermutation generation complete.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to per-transcript split file")
    args = parser.parse_args()

    print(f"Reading input data from {args.input}...", flush=True)
    mutations_df = pd.read_csv(args.input, sep='\t')

    generate_permutation_counts(
        mutations_df=mutations_df,
        output_dir="../../perm_arrays",
        n_permutations=1000000,
        verbose=True,
        by_cancer=True
    )
#%%
