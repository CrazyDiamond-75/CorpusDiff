"""
convert_to_tsv.py
Helper file to convert a Pandas Dataframe stored as a .pkl file to a .tsv.
Needs a valid .pkl as the argument.
Generates a tsv with the same name and content

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import pandas as pd
import sys

df = pd.read_pickle(sys.argv[1])

name = ""

if sys.argv[1][-4] == ".pkl":
    name = sys.argv[1][-4] + ".tsv"
else:
    name = sys.argv[1] + ".tsv"

df.to_csv(name, sep="\t")
