import pandas as pd
import sys

df = pd.read_pickle(sys.argv[1])

# Sort by strength of correlation
df.sort_values(by="Correlation", inplace=True)

df["95% CI width"] = df["95% CI"].apply(lambda x: x[1] - x[0])

for line in df.iloc:
    print(line)
