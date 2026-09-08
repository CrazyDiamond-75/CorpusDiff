"""
politweets_plot.py
Creates boxenplots, Pearson correlation coefficients, and least-squares regressor slope values based on the l-metric between NottDeuYTSch and Politweets
Needs cosine similarity scores of all labels in Politweets on NottDeuYTSch ("ndy_utf8_politweets.pkl")
Generates boxenplots under "politweets" directory, and a .pkl which stores the correlations and slope values ("ndy_politweets_correlations.pkl")

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import gc

# Theming
sns.set_theme(style="ticks", context="paper")
# Font for paper
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Crimson Text"
matplotlib.font_manager.fontManager.addfont(
    "/usr/share/fonts/CrimsonText_Regular.ttf"
)  # Add the font, such that Matplotlib can discover it

df = pd.read_pickle("ndy_utf8_politweets.pkl")

# Remove column which I don't need for plotting
df = df.drop(columns=["authorName"])

# df = df.groupby(df["publishedAt"].map(lambda x: x.year + x.month)).median()

df["publishedAt"] = df["publishedAt"].dt.to_period("Q")
df = df.rename(columns=lambda x: x.replace("freedom/", ""))


# VARIANT OF THE PLOT HERE!
# All columns which have positive and negative vectors.
dimensions_pm = [
    "constitution",
    "education",
    "europe",
    "foreign special",
    "internationalism",
    "labour",
    "military",
    "multiculturalism",
    "national way of life",
    "protectionism",
    "traditional morality",
    "welfare",
]

# Reaaaaaalllyyyy unclean code here:
for l in dimensions_pm:
    # Normalize all dimensions before calculating the difference score
    P = df[l + " +"]
    M = df[l + " -"]
    P = (P - P.mean()) / P.std()
    M = (M - M.mean()) / M.std()
    df[l] = P - M

# Just keep what we need
df = df[dimensions_pm + ["publishedAt"]]
gc.collect()


dimensions = [l for l in df.columns.array if l != "publishedAt"]

df_correlations = pd.DataFrame(
    columns=["Dimension", "Correlation", "P-Value", "95% CI", "Increase/Q"]
)

for i, l in enumerate(dimensions):
    x = df["publishedAt"]
    y = df[l]
    # medians = df.groupby("publishedAt")[l].median().to_numpy(np.float32)
    x_num = x.dt.year + (x.dt.quarter - 1) / 4.0

    # Get correlation and p-values
    res = pearsonr(x_num, y)
    Corr = res.statistic
    Pval = res.pvalue
    CInt = tuple(float(v) for v in res.confidence_interval())
    IncQ = linregress(x_num, y).slope
    df_correlations.loc[i] = [l, Corr, Pval, CInt, IncQ]

    plt.figure(figsize=(12, 6))
    # sns.lineplot(data=df, x="publishedAt", y=l, estimator="median", errorbar=("pi", 50))

    ax = sns.boxenplot(x=x, y=y, width=1.0, showfliers=False)

    # print(f"{l}\tMedian trend r = {linear_fit.rvalue:.2f}")
    # ax.set_xticks(x)
    # ax.set_xticklabels([y if y in np.arange(2008, 2019) else "" for y in x])

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"politweets/{l}.png", dpi=300)
    plt.close()

df_correlations.to_pickle("ndy_politweets_correlations.pkl")
