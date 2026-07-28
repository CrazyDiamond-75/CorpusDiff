import pandas as pd

df = pd.read_table('ndy_utf8.tsv', sep='\t')

sdf = df[['publishedAt', 'authorName', 'Text']]

# Send to GC
del df

print("CREATED DF")

sdf['publishedAt'] = sdf['publishedAt'].apply(lambda x : x[:10])
sdf['publishedAt'] = pd.to_datetime(sdf['publishedAt'], format='%Y-%m-%d')
sdf.sort_values(by='publishedAt', inplace=True)

print("SORTED SDF")

sdf = sdf.reset_index(drop=True)

sdf.to_pickle("ndy_utf8_small.pkl")

print("WROTE TO DISK")
