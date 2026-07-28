import pandas as pd

df = pd.read_pickle('ndy_utf8_small.pkl')

print(df.dtypes)
print(df)
