from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN as SkDBSCAN
from aoclda.clustering import DBSCAN as AoclDBSCAN


def dbscan_sklearn(csv_path: str, eps=0.2, min_samples=50):
    """
    Run scikit-learn DBSCAN on a CSV and save/return the cluster index arrays.
    Returns (cluster0_idx, cluster1_idx).
    """
    identifier = Path(csv_path).stem
    save_dir = Path("results/clustering_dbscan") / identifier
    save_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    X = df_raw.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    # Run DBSCAN
    db = SkDBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    labels = db.labels_.astype(int)

    # Build clusters (ignoring -1 noise)
    idx = np.arange(len(labels))
    cluster0 = idx[labels == 0]
    cluster1 = idx[labels == 1]

    # Plotting
    col = df_raw.columns[0]
    y = df_raw[col].to_numpy()

    plt.figure(figsize=(10,4))
    plt.scatter(cluster0, y[cluster0], s=8, color='blue', label="Cluster 0")
    plt.scatter(cluster1, y[cluster1], s=8, color='orange', label="Cluster 1")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.title(f"scikit-learn DBSCAN (eps={eps}, min_samples={min_samples})")
    plt.legend()
    plt.savefig(save_dir / "sklearn_dbscan_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save arrays
    np.save(save_dir / "sklearn_cluster0.npy", cluster0)
    np.save(save_dir / "sklearn_cluster1.npy", cluster1)

    return cluster0, cluster1



def dbscan_amd(csv_path: str, eps=0.2, min_samples=50):
    """
    Run AMD AOCL-DA DBSCAN on a CSV and save/return the cluster index arrays.
    Returns (cluster0_idx, cluster1_idx). Noise points (-1) are ignored.
    """

    identifier = Path(csv_path).stem
    save_dir = Path("results/clustering_dbscan") / identifier
    save_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    X = df_raw.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    # Run AMD DBSCAN
    db = AoclDBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    labels = np.asarray(db.labels, dtype=int)

    # Build clusters (ignoring -1 noise)
    idx = np.arange(len(labels))
    cluster0 = idx[labels == 0]
    cluster1 = idx[labels == 1]

    # Plotting
    col = df_raw.columns[0]
    y = df_raw[col].to_numpy()

    plt.figure(figsize=(10,4))
    plt.scatter(cluster0, y[cluster0], s=8, color='blue', label="Cluster 0")
    plt.scatter(cluster1, y[cluster1], s=8, color='orange', label="Cluster 1")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.title(f"AMD AOCL-DA DBSCAN (eps={eps}, min_samples={min_samples})")
    plt.legend()
    plt.savefig(save_dir / "amd_dbscan_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save arrays
    np.save(save_dir / "amd_cluster0.npy", cluster0)
    np.save(save_dir / "amd_cluster1.npy", cluster1)

    return cluster0, cluster1

path = r"data/regime_change/bm_n6000_GenBIN_Fr0.950_mpri15.300_spri0.400_msec1.000_ssec1.330_WarmA0.000_c0_w_0.csv"

sk0, sk1 = dbscan_sklearn(path)
amd0, amd1 = dbscan_amd(path)
