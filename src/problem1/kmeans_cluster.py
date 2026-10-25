from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as SkKMeans
from aoclda.clustering import kmeans as AoclKMeans


def kmeans_sklearn(csv_path: str):
    """
    Run scikit-learn KMeans (k=2) on a CSV and return two arrays:
    cluster0_idx, cluster1_idx
    """
    identifier = Path(csv_path).stem
    
    # Build directory path
    save_dir = Path("results/clustering_kmeans") / identifier

    # Create folder (recursive)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    X = df_raw.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    km = SkKMeans(
        n_clusters=2,
        n_init=10,
        max_iter=300,
        algorithm="lloyd",
        random_state=42,
    )
    km.fit(X)
    labels = km.labels_.astype(int)

    idx = np.arange(len(labels))
    cluster0 = idx[labels == 0]
    cluster1 = idx[labels == 1]

    # Plot 
    col = df_raw.columns[0]
    y = df_raw[col].to_numpy()

    plt.figure(figsize=(10,4))
    plt.scatter(cluster0, y[cluster0], s=8, color='blue', label="Cluster 0")
    plt.scatter(cluster1, y[cluster1], s=8, color='orange', label="Cluster 1")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.title("scikit-learn KMeans (k=2)")
    plt.legend()
    plt.savefig(save_dir / "sklearn_kmeans_clusters.png", dpi=300, bbox_inches='tight')

    np.save(save_dir / "sklearn_cluster0.npy", cluster0)
    np.save(save_dir / "sklearn_cluster1.npy", cluster1)

    return cluster0, cluster1



def kmeans_amd(csv_path: str):
    """
    Run AMD AOCL-DA KMeans (k=2) on a CSV and return two arrays:
    cluster0_idx, cluster1_idx
    """

    identifier = Path(csv_path).stem
    
    # Build directory path
    save_dir = Path("results/clustering_kmeans") / identifier

    # Create folder (recursive)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    X = df_raw.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    km = AoclKMeans(
        n_clusters=2,
        n_init=10,
        max_iter=300,
        algorithm="lloyd",
    )
    km.fit(X)
    labels = np.asarray(km.labels, dtype=int)

    idx = np.arange(len(labels))
    cluster0 = idx[labels == 0]
    cluster1 = idx[labels == 1]

    # Plot 
    col = df_raw.columns[0]
    y = df_raw[col].to_numpy()

    plt.figure(figsize=(10,4))
    plt.scatter(cluster0, y[cluster0], s=8, color='blue', label="Cluster 0")
    plt.scatter(cluster1, y[cluster1], s=8, color='orange', label="Cluster 1")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.title("AMD AOCL-DA KMeans (k=2)")
    plt.legend()
    plt.savefig(save_dir / "amd_kmeans_clusters.png", dpi=300, bbox_inches='tight')

    np.save(save_dir / "amd_cluster0.npy", cluster0)
    np.save(save_dir / "amd_cluster1.npy", cluster1)

    return cluster0, cluster1


path = r"data/regime_change/bm_n6000_GenBIN_Fr0.950_mpri15.300_spri0.400_msec1.000_ssec1.330_WarmA0.000_c0_w_0.csv"

ac0, ac1 = kmeans_amd(path)
sc0, sc1 = kmeans_sklearn(path)
