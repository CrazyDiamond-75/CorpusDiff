#!/usr/bin/env python
header = """The CorpusDiff application.
Usage: corpusdiff main_corpus diff_corpus [huggingface_model]

main_corpus and diff_corpus should be in either .csv or .tsv format.
main_corpus must have columns "Date", "Text", where each date is in "year-month-date" format.
diff_corpus must have columns "Label", "Text", where all labels end with '+' or '-' and have an inverse label.

If no hugging face model is specified, it will use T-Systems-onsite/cross-en-de-roberta-sentence-transformer
Note that this model is only useful for English and German texts.


Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT"""


def main():
    import sys

    # Parse arguments
    if len(sys.argv) not in [3, 4]:
        print(header)
        exit(1)

    print("Loading...")

    # Main imports here
    import os
    import gc
    import math
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util.similarity import cos_sim

    # Helper functions
    # Attempts to read a corpus from disk
    def read_corpus(path: str, type: bool) -> pd.DataFrame | None:
        # If type is true, the corpus is the difference corpus
        col1 = "Label" if type else "Date"

        if path.endswith(".tsv"):
            return pd.read_table(path, header=0, names=[col1, "Text"], sep="\t")
        elif path.endswith(".csv"):
            return pd.read_csv(path, header=0, names=[col1, "Text"])
        else:
            return None

    # I found 512 bytes to be ideal for my amount of vram, this method is a stepwise linear function to approximate this heuristic.
    def find_ideal_batch_size() -> int:
        vram = torch.cuda.get_device_properties(0).total_memory
        raw = int(512 * vram / 25753026560)
        return 32 * math.floor(raw / 32)

    # Same idea for 1000000 byte batch size when calculating the similarity scores.
    def find_ideal_score_batch_size() -> int:
        vram = torch.cuda.get_device_properties(0).total_memory
        raw = int(1000000 * vram / 25753026560)
        return 32 * math.floor(raw / 32)

    path_main = sys.argv[1]
    path_diff = sys.argv[2]
    model_name = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"

    # Overwrite model name to use the given one
    if len(sys.argv) == 4:
        model_name = sys.argv[3]

    model_token: str
    # Try to read a token from a hidden file
    try:
        with open(".hf_token", "r") as f:
            model_token = f.read()
    # If this fails because the file does not exist, let the user know and save the input token
    except:
        print("Please input a valid hugging face token if needed")
        model_token = input()
        if model_token not in ["", "\n", " "]:
            with open(".hf_token", "w") as f:
                f.write(model_token)
            print("Saved token to .hf_token")
        else:
            # If no "valid" token was given, use none
            model_token = None

    # Read the main corpus
    df_main: pd.DataFrame = read_corpus(path_main, False)
    if df_main is None:
        print(f"{path_main} is not a valid tsv or csv.")
        exit(1)

    # Read the difference corpus
    df_diff: pd.DataFrame = read_corpus(path_diff, True)
    if df_diff is None:
        print(f"{path_diff} is not a valid tsv or csv.")
        exit(1)

    # df_main = df_main.head()
    # df_diff = df_diff.head()
    # gc.collect()

    print(f"Loaded {path_main} and {path_diff}.")

    # Clean up the data frames and bring everything into the right format
    try:
        # Drop missing values
        df_main = df_main.drop(
            df_main[(df_main.Date == "undefined") | pd.isna(df_main.Date)].index
        )
        df_main["Date"] = pd.to_datetime(df_main["Date"], format="%Y-%m-%d")
        df_main["Text"] = df_main["Text"].map(lambda x: str(x))
        # Sort all texts in the main corpus by their date
        df_main.sort_values(by="Date", inplace=True)
        df_main.reset_index(drop=True, inplace=True)
    except:
        print(f"{path_main} is not in the right format.")
    try:
        # Drop missing values
        df_diff = df_diff.drop(
            df_diff[(df_diff.Label == "undefined") | pd.isna(df_diff.Label)].index
        )
        df_diff["Text"] = df_diff["Text"].map(lambda x: str(x))
    except:
        print(f"{path_diff} is not in the right format.")

    print(f"Cleaned up and sorted tables")

    # Convert to numpy arrays for later and force gc
    main_texts = df_main["Text"].to_numpy(str)
    del df_main["Text"]
    gc.collect()

    diff_texts = df_diff["Text"].to_numpy(str)
    del df_diff["Text"]
    gc.collect()

    # Set performance options and select cuda device if possible
    torch.set_num_threads(os.cpu_count())
    device = "cuda"
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(0)} as CUDA/ROCm device.")
    else:
        print(
            "No CUDA/ROCm device available. Maybe torch was installed wrong.\nFalling back to the CPU, which is not advised."
        )
        device = "cpu"

    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, device=device, token=model_token)

    batch_size = find_ideal_batch_size()
    print(f"Using {batch_size} byte batchsize")

    print("Encoding main corpus...")
    vectors_main = model.encode(
        main_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Convert to matrix form
    vectors_main = np.stack(vectors_main).astype(np.float32, copy=False)

    # Force gc
    torch.cuda.empty_cache()
    gc.collect()

    print("Encoding difference corpus...")
    vectors_diff = model.encode(
        diff_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # ...
    torch.cuda.empty_cache()
    gc.collect()

    print("Preprocessing labels")
    vs_by_label = {}
    for label, vector in zip(df_diff["Label"], vectors_diff):
        vs_by_label.setdefault(label, []).append(vector)

    # Convert to numerical numpy matrix
    vs_by_label = {
        label: np.stack(vectors).astype(np.float32, copy=False)
        for label, vectors in vs_by_label.items()
    }

    # Remove stuff we don't need anymore
    del vectors_diff, df_diff
    gc.collect()

    print("Moving main vectors to GPU")
    vectors_main_gpu = torch.tensor(vectors_main, dtype=torch.float32, device=device)
    # Get count of vectors (do this rather at the beginning)
    total_rows = vectors_main_gpu.shape[0]

    # Remove main vectors from memory
    del vectors_main
    gc.collect()

    print("Calculating difference scores")
    batch_size_score = find_ideal_score_batch_size()
    print(f"Using {batch_size_score} byte batchsize")

    for l, v_l in tqdm(vs_by_label.items()):
        # Move to GPU
        vectors_label_diff_gpu = torch.tensor(v_l, device=device)
        results = np.empty(total_rows, dtype=np.float32)

        for start in range(0, total_rows, batch_size_score):
            end = min(start + batch_size_score, total_rows)
            v_nd_batch = vectors_main_gpu[start:end]
            with torch.inference_mode():
                sims = cos_sim(v_nd_batch, vectors_label_diff_gpu)
                mean_sims = torch.mean(sims, dim=1)
            results[start:end] = mean_sims.cpu().numpy()
            del sims, mean_sims

        df_main[l] = results  # assign entire column at once
        del vectors_label_diff_gpu, results
        gc.collect()
        torch.cuda.empty_cache()

    print(df_main)

    exit(0)


if __name__ == "__main__":
    main()
