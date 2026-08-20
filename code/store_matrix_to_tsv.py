import pandas as pd
import gc

print("LOADING")

df = pd.read_pickle("ndy_utf8_vectors.pkl")

vectors = df["Vectors"].to_numpy()

del df
gc.collect()

print("CREATING DF")

dfv = pd.DataFrame(vectors)

del vectors
gc.collect()

name = "vector_matrix.tsv"

print("STORING")

dfv.to_csv(name, sep="\t", index=False)
