from header import *


def main():
    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_lemma.pkl")

    # print("STARTING MAP")
    print(df)


if __name__ == "__main__":
    main()
