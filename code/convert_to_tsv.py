import pandas as pd
import sys

df = pd.read_pickle(sys.argv[1])

name = ""

if sys.argv[1][-4] == ".pkl":
    name = sys.argv[1][-4] + ".tsv"
else:
    name = sys.argv[1] + ".tsv"

df.to_csv(name, sep="\t")
