from header import *

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

    parallel_map(df["Text"], count_words, 1)  # 1 Worker because of collisions

    print("SORTING")
    wc_df = pd.DataFrame(word_counts.items(), columns=["Name", "Count"])
    wc_df.sort_values(by="Count", ascending=False, inplace=True)

    wc_df.to_pickle("ndy_utf8_word_counts.pkl")


if __name__ == "__main__":
    main()
