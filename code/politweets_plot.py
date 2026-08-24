import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc

sns.set_theme(style="darkgrid")

df = pd.read_pickle("ndy_utf8_politweets.pkl")

# Remove column which I don't need for plotting
df = df.drop(columns=["authorName"])

# df = df.groupby(df["publishedAt"].map(lambda x: x.year + x.month)).median()

df["publishedAt"] = df["publishedAt"].dt.year + df["publishedAt"].dt.month / 12.0
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

# Scuffed as fuck
for l in dimensions_pm:
    df[l] = df[l + " +"] - df[l + " -"]

# Just keep what we need
df = df[dimensions_pm + ["publishedAt"]]
gc.collect()


dimensions = [l for l in df.columns.array if l != "publishedAt"]


for l in dimensions:
    # plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="publishedAt", y=l, estimator="median", errorbar=("pi", 50))
    # sns.boxenplot(data=df, x="publishedAt", y=l)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # plt.tight_layout()
    plt.savefig(f"politweets/{l}.png", dpi=300)
    plt.close()
# print(df)
