import pandas as pd
import sys

df = pd.read_pickle(sys.argv[1])

# print(df.dtypes)
# print(df)

for line in df.iloc:
    print(line)
