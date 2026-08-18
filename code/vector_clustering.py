from header import *
import gc
from sklearn.cluster import OPTICS


def main():
    print("LOADING DATASET")
    df = pd.read_pickle("ndy_utf8_vectors.pkl")

    vectors = np.stack(df["Vectors"].to_numpy()).astype(
        np.float32,
        copy=False,
    )

    del df
    gc.collect()

    # Configure the clustering algorithm to use cosine distance and use all cores.
    clusters = OPTICS(metric="cosine", n_jobs=-1)

    print("SEARCHING FOR CLUSTERS")
    clusters.fit(vectors)

    print("EXPORTING CLUSTER LABELS")
    df = Dataframe()
    df["Cluster"] = clusters.labels_
    df.to_pickle("ndy_utf8_vectors_clustered.pkl")


if __name__ == "__main__":
    main()
