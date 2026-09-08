"""
vectors_gen_politweets.py
Generates vector embeddings from the PoliTweets corpus using RoBERTa, specifically the "T-Systems-onsite/cross-en-de-roberta-sentence-transformer" model.
Needs the PoliTweets corpus in a .csv format ("politweets_v02.csv")
Generates a .pkl storing a Dataframe which contains vector embeddings as numpy arrays of float32 type and dimension 768

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from HFTOKEN import TOKEN

# from tqdm import tqdm


def main():
    print("LOADING MODEL")
    model = SentenceTransformer(
        "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        device="cuda",  # Uses ROCm if installed
        token=TOKEN,
    )
    print("MODEL LOADED")

    print("LOADING POLITWEETS")
    df = pd.read_csv("politweets_v02.csv", header=0, names=["Url", "Text", "Label"])
    # Dropping rows with "undefined" Label or missing Label.
    df = df.drop(df[(df.Label == "undefined") | pd.isna(df.Label)].index)
    print("POLITWEETS LOADED")

    print("STARTING VECTORIZATION")
    # Use tqdm to show progress over the whole column
    texts = df["Text"].to_numpy(str)
    # Remove the columns we don't need anymore
    df = df.drop(columns=["Url", "Text"])

    # torch.set_num_threads(os.cpu_count()) # If on CPU
    vectors = model.encode(
        texts,
        batch_size=512,  # Optimal for RX 7900 XTX
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Cleanup
    del texts
    torch.cuda.empty_cache()

    df["Vectors"] = list(vectors)
    df.to_pickle("politweets_v02_vectors.pkl")
    print(df)


if __name__ == "__main__":
    main()
