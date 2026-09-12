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
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr, linregress, trim_mean
from typing import Callable
import gc

# Theming
sns.set_theme(style="ticks", context="talk")
# Font for paper
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Crimson Text"
# Add the font, such that Matplotlib can discover it
matplotlib.font_manager.fontManager.addfont("/usr/share/fonts/CrimsonText_Regular.ttf")

df: pd.DataFrame = pd.read_pickle("ndy_utf8_politweets.pkl")

# Remove column which I don't need for plotting
df = df.drop(columns=["authorName"])

df["publishedAt"] = df["publishedAt"].dt.to_period("M")
df = df.rename(columns=lambda x: x.replace("freedom/", ""))


# VARIANT OF THE PLOT HERE!
# All columns which have positive and negative vectors.
# Sorted by increasing correlation for better color map
dimensions_pm = [
    "traditional morality",
    "constitution",
    "education",
    "protectionism",
    "foreign special",
    "welfare",
    "multiculturalism",
    "labour",
    "national way of life",
    "military",
    "europe",
    "internationalism",
]

palette = sns.color_palette("tab20", len(dimensions_pm))

for l in dimensions_pm:
    # Calculate l-metric
    P = df[l + " +"]
    M = df[l + " -"]
    df[l] = 0.5 * (P - M)
    # Use for normalized plot!
    # df[l] = (df[l] - df[l].mean()) / df[l].std()
# Just keep what we need
df = df[dimensions_pm + ["publishedAt"]]
gc.collect()


dimensions = [l for l in df.columns.array if l != "publishedAt"]

df_correlations = pd.DataFrame(
    columns=["Dimension", "Correlation", "P-Value", "95% CI", "Increase/Y"]
)

# Finally, plot everything in one big plot
# Make it a bit wider to fit the legend inside the ax
fig, ax = plt.subplots(figsize=(4 * 3, 4 * 2))
x = pd.PeriodIndex(df["publishedAt"]).to_timestamp()

for i, (topic, color) in enumerate(zip(dimensions, palette)):

    y = df[topic]
    x_num = (x.month - 1) + 12 * x.year

    # Get correlation and p-values
    res = pearsonr(x_num, y)
    Corr = res.statistic
    Pval = res.pvalue
    CInt = tuple(float(v) for v in res.confidence_interval())

    df_xy = pd.DataFrame({"x": x, "y": y.values})

    result = (
        df_xy.groupby("x")["y"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )

    lo = result["q25"]
    hi = result["q75"]
    y = result["median"]
    x_plt = result["x"]

    # Get median slope
    IncM = linregress(range(24097, 24226), y).slope
    df_correlations.loc[i] = [topic, Corr, Pval, CInt, IncM * 12]

    # Wow! Functional programming in Python...
    roll_filter: Callable[[pd.Series], pd.Series] = lambda x: x.rolling(
        window=12 * 2 + 1, min_periods=1, center=True
    ).apply(lambda x: trim_mean(x, 0.25))
    lo = roll_filter(lo)
    hi = roll_filter(hi)
    y = roll_filter(y)

    sns.lineplot(
        x=x_plt,
        y=y,
        ax=ax,
        color=color,
        label=topic,
    )

    # Mean of trimmed quantiles
    ax.fill_between(x=x_plt, y1=lo, y2=hi, color=color, alpha=0.15, linewidth=0.1)

# Use percentages for y-ticks
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:g}%"))
# Convert timestamp back to valid year format
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Jahr")
ax.set_ylabel("$l$-Metrik")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
ax.margins(x=0, y=0)
plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()
df_correlations.to_pickle("ndy_politweets_correlations.pkl")


print(df_correlations)
