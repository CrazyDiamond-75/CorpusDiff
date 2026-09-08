"""
load_test.py
A small helper to print out all lines in a Dataframe stored as a .pkl
Needs a valid .pkl as the runtime argument

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

# print(df.dtypes)
# print(df)

for line in df.iloc:
    print(line)
