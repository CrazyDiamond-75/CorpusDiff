import torch
from sentence_transformers.util.similarity import cos_sim
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import gc


def main():
    # Use GPU
    device = torch.device("cuda")
    torch.set_num_threads(os.cpu_count())  # If CPU is only available device

    print("LOADING NottDeuYTSch")
    df_nd = pd.read_pickle("ndy_utf8_vectors.pkl")

    v_nd = np.stack(df_nd["Vectors"].to_numpy()).astype(
        np.float32,
        copy=False,
    )

    # Move all vectors to GPU
    v_nd_gpu = torch.tensor(v_nd, dtype=torch.float32, device=device)
    total_rows = v_nd_gpu.shape[0]

    del v_nd

    # We don't need this column anymore, but for later we would like to keep the dates.
    df_nd.drop(columns=["Vectors"], inplace=True)

    gc.collect()

    print("LOADING PoliTweets")
    df_pt = pd.read_pickle("politweets_v02_vectors.pkl")

    print("PREPROCESSING")
    # Store PoliTweet vector-label pairs as dictionary.
    vs_by_label = {}
    for row in df_pt.itertuples(index=False):
        vs_by_label.setdefault(row.Label, []).append(row.Vectors)

    # Convert dict to numpy vector matrixes
    vs_by_label = {
        label: np.stack(vectors).astype(np.float32, copy=False)
        for label, vectors in vs_by_label.items()
    }

    del df_pt
    gc.collect()

    print("CALCULATING SIMILARITIES")

    # Biggest size possible with 24 Gigs VRAM
    BATCH_SIZE = 1000000

    # Idea: go through all columns (labels), and set each column as the calculated similarity score for that label.
    # For this we need to go through all labels and add a solution to the DF, when it is ready.
    # Do this in batches to prevent "heap-overflow" ;)

    for l, v_l in tqdm(vs_by_label.items()):
        # Move to GPU
        v_l_gpu = torch.tensor(v_l, device=device)
        results = np.empty(total_rows, dtype=np.float32)

        for start in range(0, total_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_rows)
            v_nd_batch = v_nd_gpu[start:end]
            with torch.inference_mode():
                sims = cos_sim(v_nd_batch, v_l_gpu)
                mean_sims = torch.mean(sims, dim=1)
            results[start:end] = mean_sims.cpu().numpy()
            del sims, mean_sims

        df_nd[l] = results  # assign entire column at once
        del v_l_gpu, results
        gc.collect()
        torch.cuda.empty_cache()

        # sims = cos_sim(v_nd, v_l)
        # mean = torch.mean(sims, 1)
        # df_nd[l] = mean

    df_nd.to_pickle("ndy_utf8_politweets.pkl")


if __name__ == "__main__":
    main()
