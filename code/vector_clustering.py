from header import *
import gc
from sklearn.cluster import KMeans


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
    # clusters = OPTICS(metric="cosine", n_jobs=-1)

    # 20 Clusters -> 20 Topic spaces. As KMeans is greedy, dense and big clusters are preferred.
    # KMeans is OK, as the vectors are normalized, thus the cosine distance is roughly equal to the euclidian metric.
    clusters = KMeans(n_clusters=20, copy_x=False)

    print("SEARCHING FOR CLUSTERS")
    clusters.fit(vectors)

    print("EXPORTING CLUSTER LABELS")
    df = pd.DataFrame()
    df["Cluster"] = clusters.labels_
    df.to_pickle("ndy_utf8_vectors_clustered.pkl")


if __name__ == "__main__":
    main()
