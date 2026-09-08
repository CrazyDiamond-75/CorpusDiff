"""
lemma.py
Converts NottDeuYTSch to a Dataframe which stores lists of lemmas instead of texts.
Needs the NottDeuYTSch corpus in .pkl format ("ndy_utf8_small.pkl")
Generates a copy of the input but with lists of lemmas ("ndy_utf8_lemma.pkl")

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import os
import pandas as pd
import spacy
from tqdm import tqdm


def main():
    print("LOADING MODEL (with only needed components)")
    # Disable parser and NER – they are not needed for lemmatisation
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    print("MODEL LOADED")

    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_small.pkl")
    print("DATASET LOADED")

    print("STARTING LEMMATISATION")
    # Use tqdm to show progress over the whole column
    texts = df["Text"].astype(str).tolist()
    lemmas = []

    with tqdm(total=len(texts), desc="Processing texts") as pbar:
        docs = nlp.pipe(texts, batch_size=64, n_process=os.cpu_count())
        for doc in docs:
            lemmas.append([token.lemma_ for token in doc if token.lemma_ != "--"])
            pbar.update(1)

    df["Text"] = lemmas
    df.to_pickle("ndy_utf8_lemma.pkl")
    print(df)


if __name__ == "__main__":
    main()
