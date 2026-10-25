import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
from pathlib import Path

# CORE ALGORITHM


def find_cutoff(
    data: np.ndarray,
    cluster0_idx: np.ndarray,
    cluster1_idx: np.ndarray,
    percentile: float = 95,
    weight: float = 0.25,
    return_details: bool = False
):
    # Determine which cluster has lower indices on average
    mean0 = np.mean(cluster0_idx)
    mean1 = np.mean(cluster1_idx)
    
    if mean0 < mean1:
        # Cluster 0 is on the left
        cutoff_left = np.percentile(cluster0_idx, percentile)
        cutoff_right = np.percentile(cluster1_idx, 100 - percentile)
    else:
        # Cluster 1 is on the left
        cutoff_left = np.percentile(cluster1_idx, percentile)
        cutoff_right = np.percentile(cluster0_idx, 100 - percentile)
    
    # Calculate weighted cutoff position
    # weight < 0.5 shifts left, weight > 0.5 shifts right
    cutoff = int(round(cutoff_left + weight * (cutoff_right - cutoff_left)))
    
    if return_details:
        details = {
            'method': 'Quantile-based',
            'cutoff_index': cutoff,
            'cutoff_left': cutoff_left,
            'cutoff_right': cutoff_right,
            'gap_size': cutoff_right - cutoff_left,
            'percentile': percentile,
            'weight': weight
        }
        return cutoff, details
    else:
        return cutoff


def evaluate_cutoff(
    cluster0_idx: np.ndarray,
    cluster1_idx: np.ndarray,
    cutoff: float
) -> Dict:

    # Determine which cluster is on which side
    mean0 = np.mean(cluster0_idx)
    mean1 = np.mean(cluster1_idx)
    
    if mean0 < mean1:
        # Cluster 0 should be before cutoff
        correctly_classified_0 = np.sum(cluster0_idx < cutoff)
        correctly_classified_1 = np.sum(cluster1_idx >= cutoff)
        misclassified_0 = np.sum(cluster0_idx >= cutoff)
        misclassified_1 = np.sum(cluster1_idx < cutoff)
    else:
        # Cluster 1 should be before cutoff
        correctly_classified_0 = np.sum(cluster0_idx >= cutoff)
        correctly_classified_1 = np.sum(cluster1_idx < cutoff)
        misclassified_0 = np.sum(cluster0_idx < cutoff)
        misclassified_1 = np.sum(cluster1_idx >= cutoff)
    
    total_correct = correctly_classified_0 + correctly_classified_1
    total_points = len(cluster0_idx) + len(cluster1_idx)
    accuracy = total_correct / total_points * 100
    
    return {
        'accuracy': accuracy,
        'cluster0_correct': correctly_classified_0,
        'cluster0_misclassified': misclassified_0,
        'cluster0_accuracy': correctly_classified_0 / len(cluster0_idx) * 100,
        'cluster1_correct': correctly_classified_1,
        'cluster1_misclassified': misclassified_1,
        'cluster1_accuracy': correctly_classified_1 / len(cluster1_idx) * 100,
        'cutoff': cutoff
    }


def classify_points(
    indices: np.ndarray,
    cutoff: float,
    return_mask: bool = False
) -> np.ndarray:

    if return_mask:
        return indices < cutoff
    else:
        return (indices >= cutoff).astype(int)

# VISUALIZATION FUNCTIONS

def visualize_cutoff(
    csv_path: str,
    cluster0_idx: np.ndarray,
    cluster1_idx: np.ndarray,
    save_path: str = "cutoff_visualization.png",
    weight: float = 0.25,
    percentile: float = 95
) -> float:

    # Load data
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 0].values
    
    # Find cutoff
    cutoff = find_cutoff(data, cluster0_idx, cluster1_idx, 
                        percentile=percentile, weight=weight)
    
    # Evaluate accuracy
    eval_result = evaluate_cutoff(cluster0_idx, cluster1_idx, cutoff)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ========== FULL VIEW ==========
    ax1 = axes[0]
    
    # Plot all points
    ax1.scatter(cluster0_idx, data[cluster0_idx], s=20, color='blue', 
               alpha=0.6, label=f'Cluster 0 (n={len(cluster0_idx)})', zorder=2)
    ax1.scatter(cluster1_idx, data[cluster1_idx], s=20, color='orange', 
               alpha=0.6, label=f'Cluster 1 (n={len(cluster1_idx)})', zorder=2)
    
    # Draw cutoff line
    ax1.axvline(cutoff, color='red', linestyle='--', linewidth=3, 
               label=f'Cutoff at index {cutoff}', zorder=3)
    
    # Shade regions
    ax1.axvspan(0, cutoff, alpha=0.08, color='blue', zorder=1)
    ax1.axvspan(cutoff, len(data), alpha=0.08, color='orange', zorder=1)
    
    # Add accuracy text
    ax1.text(0.02, 0.98, f'Accuracy: {eval_result["accuracy"]:.2f}%', 
            transform=ax1.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Cluster Separation with Quantile-Based Cutoff', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # ========== ZOOMED BOUNDARY VIEW ==========
    ax2 = axes[1]
    
    # Define zoom region
    zoom_range = 400
    zoom_start = max(0, int(cutoff - zoom_range))
    zoom_end = min(len(data), int(cutoff + zoom_range))
    
    # Filter points in zoom region
    zoom_c0_mask = (cluster0_idx >= zoom_start) & (cluster0_idx < zoom_end)
    zoom_c1_mask = (cluster1_idx >= zoom_start) & (cluster1_idx < zoom_end)
    
    zoom_c0_idx = cluster0_idx[zoom_c0_mask]
    zoom_c1_idx = cluster1_idx[zoom_c1_mask]
    
    # Plot zoomed points with better visibility
    ax2.scatter(zoom_c0_idx, data[zoom_c0_idx], s=40, color='blue', 
               alpha=0.7, label='Cluster 0', zorder=2, 
               edgecolors='darkblue', linewidths=0.5)
    ax2.scatter(zoom_c1_idx, data[zoom_c1_idx], s=40, color='orange', 
               alpha=0.7, label='Cluster 1', zorder=2,
               edgecolors='darkorange', linewidths=0.5)
    
    # Draw cutoff line
    ax2.axvline(cutoff, color='red', linestyle='--', linewidth=3, 
               label=f'Cutoff: {cutoff}', zorder=3)
    
    # Shade regions
    ax2.axvspan(zoom_start, cutoff, alpha=0.1, color='blue', zorder=1)
    ax2.axvspan(cutoff, zoom_end, alpha=0.1, color='orange', zorder=1)
    
    ax2.set_xlabel('Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title(f'Zoomed Boundary Region (±{zoom_range} indices)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.4, linestyle='--')
    ax2.set_xlim(zoom_start, zoom_end)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return cutoff


def visualize_comparison(
    csv_path: str,
    cluster0_idx: np.ndarray,
    cluster1_idx: np.ndarray,
    save_path: str = "cutoff_comparison.png",
    weight: float = 0.25
) -> float:

    df = pd.read_csv(csv_path)
    data = df.iloc[:, 0].values
    
    cutoff = find_cutoff(data, cluster0_idx, cluster1_idx, weight=weight)
    eval_result = evaluate_cutoff(cluster0_idx, cluster1_idx, cutoff)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # ========== BEFORE (Original Clustering) ==========
    ax1 = axes[0]
    ax1.scatter(cluster0_idx, data[cluster0_idx], s=10, color='blue', 
               alpha=0.6, label=f'Cluster 0 (n={len(cluster0_idx)})')
    ax1.scatter(cluster1_idx, data[cluster1_idx], s=10, color='orange', 
               alpha=0.6, label=f'Cluster 1 (n={len(cluster1_idx)})')
    ax1.set_xlabel('Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Original Clustering', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # ========== AFTER (With Cutoff Line) ==========
    ax2 = axes[1]
    ax2.scatter(cluster0_idx, data[cluster0_idx], s=10, color='blue', 
               alpha=0.6, label=f'Cluster 0')
    ax2.scatter(cluster1_idx, data[cluster1_idx], s=10, color='orange', 
               alpha=0.6, label=f'Cluster 1')
    
    # Draw cutoff line
    ax2.axvline(cutoff, color='red', linestyle='--', linewidth=3, 
               label=f'Cutoff: {cutoff}', zorder=10)
    
    # Shade regions
    ax2.axvspan(0, cutoff, alpha=0.1, color='blue', label='← Cluster 0 side')
    ax2.axvspan(cutoff, len(data), alpha=0.1, color='orange', label='Cluster 1 side →')
    
    ax2.set_xlabel('Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title(f'With Cutoff Line\nAccuracy: {eval_result["accuracy"]:.1f}%', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return cutoff


def visualize_core_outliers(
    csv_path: str,
    cluster0_idx: np.ndarray,
    cluster1_idx: np.ndarray,
    save_path: str = "cutoff_core_outliers.png",
    weight: float = 0.25,
    percentile: float = 95
) -> float:

    # Load data
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 0].values
    
    # Find cutoff
    cutoff = find_cutoff(data, cluster0_idx, cluster1_idx, 
                        percentile=percentile, weight=weight)
    
    # Identify core vs outlier points
    outlier_threshold_c0 = np.percentile(cluster0_idx, percentile)
    outlier_threshold_c1 = np.percentile(cluster1_idx, 100 - percentile)
    
    core_c0 = cluster0_idx[cluster0_idx <= outlier_threshold_c0]
    outlier_c0 = cluster0_idx[cluster0_idx > outlier_threshold_c0]
    
    core_c1 = cluster1_idx[cluster1_idx >= outlier_threshold_c1]
    outlier_c1 = cluster1_idx[cluster1_idx < outlier_threshold_c1]
    
    # Evaluate accuracy
    eval_result = evaluate_cutoff(cluster0_idx, cluster1_idx, cutoff)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot core points (solid)
    ax.scatter(core_c0, data[core_c0], s=25, color='darkblue', 
              alpha=0.7, label=f'Cluster 0 Core (n={len(core_c0)})', zorder=3)
    ax.scatter(core_c1, data[core_c1], s=25, color='darkorange', 
              alpha=0.7, label=f'Cluster 1 Core (n={len(core_c1)})', zorder=3)
    
    # Plot outliers (marked with X)
    if len(outlier_c0) > 0:
        ax.scatter(outlier_c0, data[outlier_c0], s=80, color='blue', 
                  alpha=0.4, marker='x', linewidths=2,
                  label=f'Cluster 0 Outliers (n={len(outlier_c0)})', zorder=2)
    
    if len(outlier_c1) > 0:
        ax.scatter(outlier_c1, data[outlier_c1], s=80, color='orange', 
                  alpha=0.4, marker='x', linewidths=2,
                  label=f'Cluster 1 Outliers (n={len(outlier_c1)})', zorder=2)
    
    # Draw cutoff line
    ax.axvline(cutoff, color='red', linestyle='--', linewidth=3, 
              label=f'Cutoff at index {cutoff}', zorder=4)
    
    # Add accuracy text
    ax.text(0.02, 0.98, 
           f'Accuracy: {eval_result["accuracy"]:.2f}%\n'
           f'Core: {len(core_c0) + len(core_c1)} points\n'
           f'Outliers: {len(outlier_c0) + len(outlier_c1)} points', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Separation: Core vs Outliers', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return cutoff


# EXAMPLE USAGE & MAIN

def print_summary(cluster0_idx, cluster1_idx, cutoff):
    """Print summary of cutoff analysis."""
    eval_result = evaluate_cutoff(cluster0_idx, cluster1_idx, cutoff)
    
    print("\n" + "="*70)
    print("CLUSTER CUTOFF ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total points: {len(cluster0_idx) + len(cluster1_idx)}")
    print(f"Cluster 0: {len(cluster0_idx)} points")
    print(f"Cluster 1: {len(cluster1_idx)} points")
    print("-"*70)
    print(f"Cutoff Index: {cutoff}")
    print(f"Overall Accuracy: {eval_result['accuracy']:.2f}%")
    print("-"*70)
    print(f"Cluster 0:")
    print(f"  - Correctly classified: {eval_result['cluster0_correct']}/{len(cluster0_idx)}")
    print(f"  - Accuracy: {eval_result['cluster0_accuracy']:.2f}%")
    print(f"  - Misclassified: {eval_result['cluster0_misclassified']}")
    print(f"\nCluster 1:")
    print(f"  - Correctly classified: {eval_result['cluster1_correct']}/{len(cluster1_idx)}")
    print(f"  - Accuracy: {eval_result['cluster1_accuracy']:.2f}%")
    print(f"  - Misclassified: {eval_result['cluster1_misclassified']}")
    print("="*70 + "\n")


def demo():
   
    print("\n" + "="*80)
    print("="*80 + "\n")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    csv_file = "bm_n2525_GenBIN_Fr1.000_mpri12.300_spri0.340_msec0.000_ssec0.000_WarmA2.000_c200_w_200.csv"
    csv_stem = Path(csv_file).stem  # removes .csv

    csvPath = PROJECT_ROOT / "data" / "regime_change" / csv_file

    clusters_dir = PROJECT_ROOT / "results" / "clustering_kmeans" / csv_stem
    cluster0_path = clusters_dir / "amd_cluster0.npy"
    cluster1_path = clusters_dir / "amd_cluster1.npy"

    print("csvPath =", csvPath)
    print("cluster0_path =", cluster0_path)
    print("cluster1_path =", cluster1_path)

    if not cluster0_path.exists() or not cluster1_path.exists():
        raise FileNotFoundError(
            "Cluster files not found for this CSV.\n"
            f"Expected:\n  {cluster0_path}\n  {cluster1_path}\n"
            "This usually means the clustering step didn’t save results for this dataset, "
            "or they were saved under a different folder name."
        )

    cluster0_idx = np.load(cluster0_path, allow_pickle=True)
    cluster1_idx = np.load(cluster1_path, allow_pickle=True)

    df = pd.read_csv(csvPath)
    data = df.iloc[:, 0].values

    n = len(data)
    max_idx = max(int(cluster0_idx.max()), int(cluster1_idx.max()))
    if max_idx >= n:
        raise ValueError(
            f"Mismatch: CSV has {n} points but clusters contain index {max_idx}.\n"
            f"CSV: {csv_stem}\nClusters folder: {clusters_dir}"
        )
    
    # Find cutoff
    cutoff, details = find_cutoff(data, cluster0_idx, cluster1_idx, 
                                  return_details=True)
    
    print(f"\n   Method: {details['method']}")
    print(f"   Cutoff: {details['cutoff_index']}")
    print(f"   Gap: [{details['cutoff_left']:.1f}, {details['cutoff_right']:.1f}]")
    print(f"   Gap size: {details['gap_size']:.1f} indices")
    print(f"   Weight: {details['weight']}")
    
    # Print summary
    print_summary(cluster0_idx, cluster1_idx, cutoff)
    
    # Generate visualizations
    
    visualize_cutoff(csvPath, cluster0_idx, cluster1_idx,
                    save_path="cutoff_visualization.png")
    
    visualize_comparison(csvPath, cluster0_idx, cluster1_idx,
                        save_path="cutoff_comparison.png")
    
    visualize_core_outliers(csvPath, cluster0_idx, cluster1_idx,
                           save_path="cutoff_core_outliers.png")
    
    print("\nAll visualizations saved!")
    print(" cutoff_visualization.png  - Main visualization")
    print(" cutoff_comparison.png     - Before/after comparison")
    print(" cutoff_core_outliers.png  - Core vs outliers\n")
 


if __name__ == "__main__":
    print(__doc__)
    demo()
