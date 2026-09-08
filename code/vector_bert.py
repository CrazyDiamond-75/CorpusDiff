"""
vector_bert.py
Applies BERTopic on NottDeuYTSch
Needs:
- vector embeddings ("ndy_utf8_vectors.pkl")
- texts ("ndy_utf8_small.pkl")
Generates:
- Topic Identifier for Sentence ("ndy_utf8_BERT_topics.pkl")
- Topic Name for Topic Identifier ("ndy_utf8_BERT_topic_names.pkl")

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
import gc
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    metric="cosine",
    low_memory=True,
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=100,
    min_samples=10,
    prediction_data=True,
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,  # Else it just takes too long sadly :(
    verbose=True,
    language="german",
)


def main():
    print("LOADING VECTORS")
    df = pd.read_pickle("ndy_utf8_vectors.pkl")

    vectors = np.stack(df["Vectors"].to_numpy())

    del df
    gc.collect()

    print("LOADING SENTENCES")
    df = pd.read_pickle("ndy_utf8_small.pkl")

    sentences = df["Text"].astype(str).to_numpy()
    del df
    gc.collect()

    print("FITTING BERTOPIC")
    topics, probabilities = topic_model.fit_transform(sentences, vectors)

    print("SAVING TOPICS")
    df = pd.DataFrame()
    df["Topics"] = topics
    df["Confidence"] = probabilities

    df.to_pickle("ndy_utf8_BERT_topics.pkl")

    print("SAVING TOPIC NAMES")
    topic_info = topic_model.get_topic_info()  # columns: Topic, Count, Name
    topic_info.to_pickle("ndy_utf8_BERT_topic_names.pkl")


if __name__ == "__main__":
    main()
