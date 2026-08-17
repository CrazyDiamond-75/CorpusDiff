from header import *


def main():
    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_vectors.pkl")

    print(df['Vectors'].head())


if __name__ == "__main__":
    main()
