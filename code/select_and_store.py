"""
select_and_store.py
Removes unnecessary information from NottDeuYTSch and stores it as a .pkl
Needs NottDeuYTSch in TSV format ("ndy_utf8.tsv")
Generates .pkl storing the needed columns of NottDeuYTSch ("ndy_utf8_small.pkl")

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import pandas as pd

df = pd.read_table("ndy_utf8.tsv", sep="\t")

sdf = df[["publishedAt", "authorName", "Text"]]

# Send to GC
del df

print("CREATED DF")

sdf["publishedAt"] = sdf["publishedAt"].apply(lambda x: x[:10])
sdf["publishedAt"] = pd.to_datetime(sdf["publishedAt"], format="%Y-%m-%d")
sdf["Text"] = sdf["Text"].map(lambda x: str(x))  # Force convert to string
sdf.sort_values(by="publishedAt", inplace=True)

print("SORTED SDF")

sdf = sdf.reset_index(drop=True)

sdf.to_pickle("ndy_utf8_small.pkl")

print("WROTE TO DISK")
