"""
gen_word_counts.py
Counts each lemma in NottDeuYTSch
Needs lemmas generated from NottDeuYTSch ("ndy_utf8_lemma.pkl")
Generates a Dataframe which stores the counts of each lemma ("ndy_utf8_word_counts.pkl")

Copyright 2026 by
Henri Heyden

This program and the accompanying materials are made
available under the terms of the MIT License which
is available at https://opensource.org/license/MIT.

SPDX-License-Identifier: MIT
"""

import pandas as pd

# Static global for latter use
word_counts = {}


def count_words(line):
    global word_counts

    # Line is an array of lemmatized words
    for word in line:
        word = word.casefold()

        # print(word)
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1


def main():
    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_lemma.pkl")

    print("STARTING MAP")

    df["Text"].map(count_words)

    print("SORTING")
    wc_df = pd.DataFrame(word_counts.items(), columns=["Name", "Count"])
    wc_df.sort_values(by="Count", ascending=False, inplace=True)

    wc_df.to_pickle("ndy_utf8_word_counts.pkl")


if __name__ == "__main__":
    main()
