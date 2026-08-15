import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from HFTOKEN import TOKEN

# from tqdm import tqdm


def main():
    print("LOADING MODEL")
    # "de_core_news_sm" (Spacy Model in case of fallback)
    model = SentenceTransformer(
        "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        device="cuda",  # Uses ROCm if installed
        token=TOKEN,
    )
    print("MODEL LOADED")

    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_small.pkl")
    print("DATASET LOADED")

    print("STARTING VECTORIZATION")
    # Use tqdm to show progress over the whole column
    texts = df["Text"].astype(str).tolist()
    # Remove the column
    df = df.drop(columns=["Text"])

    """
    with tqdm(total=len(texts), desc="Processing texts") as pbar:
        docs = nlp.pipe(texts, batch_size=64, n_process=os.cpu_count())
        for doc in docs:
            vectors.append(doc.vector)
            pbar.update(1)
    """

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
