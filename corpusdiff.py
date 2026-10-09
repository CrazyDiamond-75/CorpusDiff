#!/usr/bin/env python
header = """The CorpusDiff application.
Usage: corpusdiff.py main_corpus diff_corpus [huggingface_model]

main_corpus and diff_corpus should be in either .csv or .tsv format.
main_corpus must have columns "Date", "Text", where each date is in "year-month-date" format.
diff_corpus must have columns "Label", "Text", where all labels end with ' +' or ' -' and have an inverse label.

If no hugging face model is specified, it will use T-Systems-onsite/cross-en-de-roberta-sentence-transformer
Note that this model is only useful for English and German texts.


Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT"""

print("Loading imports...")

import gc
import math
import os
import pickle
import sys
from typing import Callable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress, pearsonr, trim_mean
from sentence_transformers import SentenceTransformer
from sentence_transformers.util.similarity import cos_sim
from tqdm import tqdm


def read_corpus(path: str, is_diff_corpus: bool) -> Optional[pd.DataFrame]:
    """Attempts to read a corpus from disk."""
    col1 = "Label" if is_diff_corpus else "Date"

    if path.endswith(".tsv"):
        return pd.read_table(path, header=0, names=[col1, "Text"], sep="\t")
    elif path.endswith(".csv"):
        return pd.read_csv(path, header=0, names=[col1, "Text"])
    else:
        return None


def find_ideal_batch_size(my_batch: int, min_batch: int) -> int:
    """Stepwise linear approximation of ideal batch size based on VRAM."""
    vram = torch.cuda.get_device_properties(0).total_memory
    raw = my_batch * vram / 25753026560
    return max(min_batch, 32 * math.floor(raw / 32))


def remove_duplicates(lst: list) -> list:
    """Removes duplicates from a sorted list."""
    new_lst = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i - 1] != lst[i]:
            new_lst.append(lst[i])
    return new_lst


def parse_args(argv: List[str]) -> Tuple[str, str, str]:
    if len(argv) not in [3, 4]:
        print(header)
        sys.exit(1)

    path_main = argv[1]
    path_diff = argv[2]
    model_name = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"

    if len(argv) == 4:
        model_name = argv[3]

    return path_main, path_diff, model_name


def get_model_token() -> Optional[str]:
    try:
        with open(".hf_token", "r") as f:
            model_token = f.read().strip()
            if not model_token:
                print(".hf_token does not contain a token")
                model_token = None
    except FileNotFoundError:
        print("Please input a valid hugging face token if needed")
        model_token = input().strip()
        if model_token:
            with open(".hf_token", "w") as f:
                f.write(model_token)
            print("Saved token to .hf_token")
        else:
            print("Using no token")
            model_token = None

    return model_token


def clean_main_corpus(df_main: pd.DataFrame, path_main: str) -> pd.DataFrame:
    try:
        df_main = df_main[~((df_main.Date == "undefined") | df_main.Date.isna())]
        df_main = df_main[~((df_main.Text == "undefined") | df_main.Text.isna())]

        df_main["Date"] = pd.to_datetime(df_main["Date"], format="%Y-%m-%d")
        df_main["Date"] = df_main["Date"].dt.to_period("M")
        df_main["Text"] = df_main["Text"].astype(str)
        df_main = df_main.sort_values(by="Date").reset_index(drop=True)
    except (pd.errors.ParserError, ValueError):
        print(
            f"Dates in {path_main} are not in the right format, they should be in \"%Y-%m-%d\"."
        )
        sys.exit(1)

    return df_main


def clean_diff_corpus(df_diff: pd.DataFrame) -> pd.DataFrame:
    df_diff = df_diff[~((df_diff.Label == "undefined") | df_diff.Label.isna())]
    df_diff = df_diff[~((df_diff.Text == "undefined") | df_diff.Text.isna())]
    df_diff["Text"] = df_diff["Text"].astype(str)
    return df_diff


def setup_device() -> str:
    torch.set_num_threads(os.cpu_count())
    device = "cuda"

    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(0)} as CUDA/ROCm device.")
    else:
        print(
            "No CUDA/ROCm device available. Maybe torch was installed wrong.\n"
            "Falling back to the CPU, which is not advised."
        )
        device = "cpu"

    return device


def compute_lmetrics(
    path_main: str,
    path_diff: str,
    model_name: str,
    model_token: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    # Read the main corpus
    df_main = read_corpus(path_main, False)
    if df_main is None:
        print(f"{path_main} is not a valid tsv or csv.")
        sys.exit(1)

    # Read the difference corpus
    df_diff = read_corpus(path_diff, True)
    if df_diff is None:
        print(f"{path_diff} is not a valid tsv or csv.")
        sys.exit(1)

    print(f"Loaded {path_main} and {path_diff}.")

    df_main = clean_main_corpus(df_main, path_main)
    df_diff = clean_diff_corpus(df_diff)

    print("Cleaned up and sorted tables")

    main_texts = df_main["Text"].to_numpy(str)
    del df_main["Text"]
    gc.collect()

    diff_texts = df_diff["Text"].to_numpy(str)
    del df_diff["Text"]
    gc.collect()

    device = setup_device()

    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, device=device, token=model_token)

    batch_size = find_ideal_batch_size(512, 32)
    print(f"Using {batch_size} byte batchsize")

    print("Encoding main corpus...")
    vectors_main = model.encode(
        main_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors_main = np.stack(vectors_main).astype(np.float32, copy=False)

    torch.cuda.empty_cache()
    gc.collect()

    print("Encoding difference corpus...")
    vectors_diff = model.encode(
        diff_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    torch.cuda.empty_cache()
    gc.collect()

    print("Preprocessing labels")
    vs_by_label = {}
    for label, vector in zip(df_diff["Label"], vectors_diff):
        vs_by_label.setdefault(label, []).append(vector)

    # Convert from lists of vectors to vector matrices
    vs_by_label = {
        label: np.stack(vectors).astype(np.float32, copy=False)
        for label, vectors in vs_by_label.items()
    }

    del vectors_diff, df_diff
    gc.collect()

    print("Moving main vectors to GPU")
    vectors_main_gpu = torch.tensor(vectors_main, dtype=torch.float32, device=device)
    total_rows = vectors_main_gpu.shape[0]

    del vectors_main
    gc.collect()

    print("Calculating difference scores")
    batch_size_score = find_ideal_batch_size(1000000, 512)
    print(f"Using {batch_size_score} byte batchsize")

    new_cols = {}
    for l, v_l in tqdm(vs_by_label.items()):
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

        new_cols[l] = results
        del vectors_label_diff_gpu, results
        gc.collect()
        torch.cuda.empty_cache()

    df_main = pd.concat([df_main, pd.DataFrame(new_cols, index=df_main.index)], axis=1)

    dimensions = sorted(list({l[:-2] for l in vs_by_label.keys()}))

    print("Calculating l-metrics")
    for l in dimensions:
        P = df_main[l + " +"]
        M = df_main[l + " -"]
        df_main[l] = 0.5 * (P - M)

        del P, M
        gc.collect()

    df_main = df_main[["Date"] + dimensions]
    gc.collect()

    print("Saving calculated l-metrics to disk")
    df_main.to_pickle("lmetrics.pkl")

    return df_main, dimensions


def load_or_compute_lmetrics(
    path_main: str,
    path_diff: str,
    model_name: str,
    model_token: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    try:
        df_main = pd.read_pickle("lmetrics.pkl")
        dimensions = df_main.columns[1:]  # Remove "Date"
    except (pickle.UnpicklingError, FileNotFoundError):
        df_main, dimensions = compute_lmetrics(
            path_main, path_diff, model_name, model_token
        )

    return df_main, dimensions


def plot_and_correlate(df_main: pd.DataFrame, dimensions: List[str]) -> None:
    df_correlations = pd.DataFrame(
        columns=["Dimension", "Correlation", "P-Value", "95% CI", "Increase/Y"]
    )

    print("Calculating correlations, increase per year, and plotting...")

    sns.set_theme(style="ticks", context="talk")
    palette = sns.color_palette("tab20", len(dimensions))
    fig, ax = plt.subplots(figsize=(4 * 3, 4 * 2))

    X = pd.PeriodIndex(df_main["Date"]).to_timestamp()

    for i, (topic, color) in tqdm(enumerate(zip(dimensions, palette))):
        Y = df_main[topic]
        X_num = (X.month - 1) + 12 * X.year

        res = pearsonr(X_num, Y)
        Corr = res.statistic
        Pval = res.pvalue
        CInt = tuple(float(v) for v in res.confidence_interval())

        df_xy = pd.DataFrame({"X": X, "Y": Y.values})

        result = (
            df_xy.groupby("X")["Y"]
            .agg(
                median="median",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75),
            )
            .reset_index()
        )

        lo = result["q25"]
        hi = result["q75"]
        Y = result["median"]
        X_plt = result["X"]

        X_num_no_dup = remove_duplicates(X_num)

        IncM = linregress(X_num_no_dup, Y).slope
        df_correlations.loc[i] = [topic, Corr, Pval, CInt, IncM * 12]

        roll_filter: Callable[[pd.Series], pd.Series] = lambda x: x.rolling(
            window=12 * 2 + 1, min_periods=1, center=True
        ).apply(lambda x: trim_mean(x, 0.25))
        lo = roll_filter(lo)
        hi = roll_filter(hi)
        Y = roll_filter(Y)

        sns.lineplot(
            x=X_plt,
            y=Y,
            ax=ax,
            color=color,
            label=topic,
        )

        ax.fill_between(x=X_plt, y1=lo, y2=hi, color=color, alpha=0.15, linewidth=0.1)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:g}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Year")
    ax.set_ylabel("$l$-metric")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.margins(x=0, y=0)
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

    print(df_correlations.to_string())


def main() -> None:
    path_main, path_diff, model_name = parse_args(sys.argv)

    model_token = get_model_token()
    df_main, dimensions = load_or_compute_lmetrics(
        path_main, path_diff, model_name, model_token
    )
    plot_and_correlate(df_main, dimensions)

    sys.exit(0)


if __name__ == "__main__":
    main()
