import re
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configure tqdm to use pandas
tqdm.pandas(desc="Processing")


# Version of df.map which splits df into chunks for each core to take.
def parallel_map(df, func, workers=None):
    workers = workers or os.cpu_count()
    print(f"USING {workers} WORKERS")
    # split by integer positions so we can take Series slices (preserves .map)
    indices = np.array_split(np.arange(len(df)), workers)

    def apply_idx(idx):
        return df.iloc[idx].progress_map(func)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(apply_idx, indices))

    return pd.concat(results)


def do(funcs, x):
    for f in funcs:
        f(x)


def nothing(_):
    pass


word_counts = {}


def count_words(line):
    global word_counts
    # Use a tokenizer to tokenize text.
    for word in line:
        # print(word)
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1


def main():
    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_lemma.pkl")
    print("LOADING DATASET COMPLETE")

    # print(df)

    print("STARTING MAP")

    parallel_map(df["Text"], count_words, 1)  # 1 Worker because of collisions
    # print(word_counts)
    words = list(word_counts.keys())
    words.sort(key=lambda x: -word_counts[x])
    print(words[:50])


if __name__ == "__main__":
    main()
