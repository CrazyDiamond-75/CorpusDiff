from header import *
from sklearn.decomposition import PCA


def main():
    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_vectors.pkl")

    vectors = np.stack(df["Vectors"].to_numpy())
    del df

    print("PERFORMING PCA")
    pca = PCA()
    pca.fit(vectors)

    covr = pca.explained_variance_ratio_
    covr.sort()

    # From these results, we get that the information density is too high, that applying PCA would really make a difference, thus for now, don't change anything.
    print(f"COV-RATIOS: {covr}")


if __name__ == "__main__":
    main()
