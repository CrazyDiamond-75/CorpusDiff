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

def parallel_apply(df, func, workers=None):
    workers = workers or os.cpu_count()
    print(f"USING {workers} WORKERS")
    # split by integer positions so we can take Series slices (preserves .map)
    indices = np.array_split(np.arange(len(df)), workers)

    def apply_idx(idx):
        return df.iloc[idx].progress_apply(func)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(apply_idx, indices))

    return pd.concat(results)


def do(funcs, x):
    for f in funcs:
        f(x)


def nothing(_):
    pass
