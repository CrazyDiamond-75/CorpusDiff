"""
clean_correlations.py
Prints out l-metric correlations and their 95% CI width
Needs correlations ("ndy_politweets_correlations.pkl")

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import pandas as pd

df = pd.read_pickle("ndy_politweets_correlations.pkl")

# Sort by strength of correlation
df.sort_values(by="Correlation", inplace=True)

df["95% CI width"] = df["95% CI"].apply(lambda x: x[1] - x[0])

for line in df.iloc:
    print(line)
