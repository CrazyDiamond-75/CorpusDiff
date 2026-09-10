"""
bertopic_plot.py
Creates plots from "interesting" topics which were found with BERTopic
Needs both BERTopic outputs ("ndy_utf8_BERT_topics_names.pkl", "ndy_utf8_BERT_topics.pkl"), and the small NottDeuYTSch version ("ndy_utf8_small.pkl")

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import matplotlib
import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from scipy.stats import trim_mean
import seaborn as sns
import pandas as pd
import re

# Theming
sns.set_theme(style="ticks", context="talk")
# Font for paper
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Crimson Text"
# Add the font, such that Matplotlib can discover it
matplotlib.font_manager.fontManager.addfont("/usr/share/fonts/CrimsonText_Regular.ttf")

palette = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
]

df_names = pd.read_pickle("ndy_utf8_BERT_topics_names.pkl")

# Convert to dict mapping topic to topic name
names = dict(zip(df_names["Topic"], df_names["Name"]))
del df_names

df_topics = pd.read_pickle("ndy_utf8_BERT_topics.pkl")
df_topics = df_topics.drop(
    columns=["Confidence"]
)  # The confidence is flawed, as we didn't use `calculate_probabilities=True`

df_ndy = pd.read_pickle("ndy_utf8_small.pkl")
df_ndy = df_ndy.drop(
    columns=["authorName", "Text"]
)  # We only need the "publishedAt"-key for binning!

# Important fail-safe if something goes wrong!
# Every comment MUST HAVE a topic assigned!
assert len(df_topics) == len(df_ndy)
assert df_topics.index.equals(df_ndy.index)

# Join both data frames
df = pd.concat([df_ndy, df_topics], axis=1, join="outer")

del df_topics
del df_ndy

# Group by month
df["publishedAt"] = df["publishedAt"].dt.to_period("M")

interesting_topics = set([7, 10, 12, 14, 43, 47])  # As specified in Table 2
# interesting_topics = set([12, 14, 43, 47])  # As specified in Table 2

# Initialize empty 2d dictionary which stores counts for month for every topic
counts_pq_pt = {t: dict() for t in interesting_topics}

# Map each month to comment count to normalize each count
c_count_pq = dict()

for line in df.itertuples():
    month = line[1]
    topic = int(line[2])

    if month not in c_count_pq.keys():
        c_count_pq[month] = 1
    else:
        c_count_pq[month] += 1

    # Filter out unneeded topic entries
    if not topic in interesting_topics:
        continue

    if month not in counts_pq_pt[topic]:
        counts_pq_pt[topic][month] = 1
    else:
        counts_pq_pt[topic][month] += 1


# Normalize each count to get relative occurrences
for topic in interesting_topics:
    for month in c_count_pq.keys():
        if month not in counts_pq_pt[topic].keys():
            counts_pq_pt[topic][month] = 0
        else:
            counts_pq_pt[topic][month] /= float(c_count_pq[month])

# Finally, plot everything in one big plot
fig, ax = plt.subplots(figsize=(4 * 2, 4 * 2))

months = sorted(c_count_pq.keys())
x = pd.PeriodIndex(months).to_timestamp()

for topic, color in zip(interesting_topics, palette):
    series = pd.Series(counts_pq_pt[topic]).sort_index()

    # 24-month (1 year) passing-window median to reduce outlier (singular videos about a certain topic) influence
    series = series.rolling(window=12 * 2 + 1, min_periods=1, center=True).apply(
        lambda x: trim_mean(x, 0.25)
    )

    # Remove number and first "_", trim to fit in legend.
    shortname = re.sub("[0-9]+_", "", names[topic][:10]) + "..."

    sns.lineplot(x=x, y=series.values, ax=ax, label=shortname, color=color)

# Log the scale for better visuals
ax.set_yscale("symlog", base=10, linthresh=0.0025)

# Use percentages for y-ticks
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:g}%"))
# Convert timestamp back to valid year format
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Jahr")
ax.set_ylabel("Relative Häufigkeit")
ax.legend()
ax.margins(x=0, y=0)
plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
