"""
vectors_gen_nottdeuytsch.py
Generates vector embeddings from the NottDeuYTSch corpus using RoBERTa, specifically the "T-Systems-onsite/cross-en-de-roberta-sentence-transformer" model.
Needs the adjusted NottDeuYTSch version ("ndy_utf8_small.pkl")
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

    print("LOADING NOTTDEUYTSCH")
    df = pd.read_pickle("ndy_utf8_small.pkl")
    print("NOTTDEUYTSCH LOADED")

    print("STARTING VECTORIZATION")
    # Use tqdm to show progress over the whole column
    texts = df["Text"].to_numpy(str)
    # Remove the column we don't need anymore
    df = df.drop(columns=["Text"])

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
    df.to_pickle("ndy_utf8_vectors.pkl")
    print(df)


if __name__ == "__main__":
    main()
