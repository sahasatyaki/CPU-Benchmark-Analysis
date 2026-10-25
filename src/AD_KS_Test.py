import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, anderson_ksamp
from pathlib import Path
import sys


def kolmogorov_smirnov_test(file1_path: str, file2_path: str, window_size: int = 100): 
    cluster1 = np.load(file1_path)
    cluster2 = np.load(file2_path)
    
    print("KOLMOGOROV-SMIRNOV TEST")
    print(f"Cluster 1 consists of: {len(cluster1)} indices") #number of points in the first cluster
    print(f"Cluster 2 consists of: {len(cluster2)} indices") #number of points in the second cluster
    ks_statistic, ks_pvalue = ks_2samp(cluster1, cluster2)
    print(f"\nKS Statistic: {ks_statistic:.6f}")
    print(f"P-value: {ks_pvalue:.6e}") #the probability value
    
    if ks_pvalue < 0.05:
        print(f"\nReject null hypothesis (p < 0.05)")
        print(f"The two clusters come from different distributions!")
    else:
        print(f"\nCannot reject null hypothesis (p >= 0.05)")
        print(f"The two clusters come from similar/same distributions!")
    
    change_point = None #index where the largest distribution change occurs
    max_ks = None #max KS value across the all sliding windows/essentially the strength of the detected change if any
    ks_scores = []
    positions = [] #time index corresponding to each ks score

    #in order to convert the indices from both the .npy files into 1 for time series analysis
    temporal_sequence = None #0:index belongs to cluster 1, 1:index belongs to cluster 2, 2:index belongs to both clusters, -1: index is in neithwer
    time_points = None #cluster indices might start at different point not just 0.
    
    if ks_pvalue < 0.05:
        # Change Point Detection
        print(f"Change Point (window size = {window_size}):")
        
        #full span over which the clusters exists, again since its not necessary that cluster will start at 0 
        all_indices = np.concatenate([cluster1, cluster2])
        min_idx = int(all_indices.min())
        max_idx = int(all_indices.max())
        
        temporal_sequence = []
        for idx in range(min_idx, max_idx + 1): #ensuring that the time is continuous and not broken
            in_cluster1 = idx in cluster1
            in_cluster2 = idx in cluster2
            
            #in order to again maintain temporal continuity
            if in_cluster1 and in_cluster2:
                temporal_sequence.append(2)  
            elif in_cluster1:
                temporal_sequence.append(0)  
            elif in_cluster2:
                temporal_sequence.append(1)  
            else:
                temporal_sequence.append(-1)  
        
        temporal_sequence = np.array(temporal_sequence)
        time_points = np.arange(min_idx, max_idx + 1)
        
        ks_scores = []
        positions = []

        #If I split the time series here, does the statistics of clusters change
        for i in range(len(temporal_sequence) - 2 * window_size):
            before_window = temporal_sequence[i:i+window_size] #assuming that the neighboring indices represent adjacent times
            after_window = temporal_sequence[i+window_size:i+2*window_size]
            
            before_vals = before_window[before_window >= 0]
            after_vals = after_window[after_window >= 0]

            if len(before_vals) > 10 and len(after_vals) > 10:
                ks_stat, _ = ks_2samp(before_vals, after_vals)
                ks_scores.append(ks_stat)
                positions.append(time_points[i + window_size])  
        
        if len(ks_scores) > 0:
            ks_scores = np.array(ks_scores)
            positions = np.array(positions)
            
            change_idx = np.argmax(ks_scores)
            change_point = positions[change_idx]
            max_ks = ks_scores[change_idx]
            
            print(f"Change point detected at index: {int(change_point)}")
            print(f"Maximum KS statistic: {max_ks:.6f}")
        else:
            print("Not enough data points for change point detection")
    
    if ks_pvalue < 0.05:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Plot 1: Temporal sequence
        ax1 = axes[0]
        colors = ['blue', 'orange', 'purple', 'gray']
        
        for val, color, label in zip([0, 1, 2, -1], colors, ['Cluster 1', 'Cluster 2', 'Both', 'Neither']):
            mask = temporal_sequence == val
            if mask.any():
                ax1.scatter(time_points[mask], temporal_sequence[mask], c=color, label=label, alpha=0.6, s=10)
        
        if change_point is not None:
            ax1.axvline(change_point, color='red', linestyle='--', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
        
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Cluster Membership')
        ax1.set_title('Temporal Sequence of Cluster Membership')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['C1', 'C2', 'Both'])
        
        # Plot 2: Histogram
        ax2 = axes[1]
        bins = np.linspace(min(cluster1.min(), cluster2.min()), max(cluster1.max(), cluster2.max()), 50)
        ax2.hist(cluster1, bins=bins, alpha=0.6, label='Cluster 1', color='blue', density=True)
        ax2.hist(cluster2, bins=bins, alpha=0.6, label='Cluster 2', color='orange', density=True)
        
        if change_point is not None:
            ax2.axvline(change_point, color='red', linestyle='--', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
        
        ax2.set_xlabel('Index Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: CDFs
        ax3 = axes[2]
        sorted_cluster1 = np.sort(cluster1)
        sorted_cluster2 = np.sort(cluster2)
        cdf1 = np.arange(1, len(sorted_cluster1) + 1) / len(sorted_cluster1)
        cdf2 = np.arange(1, len(sorted_cluster2) + 1) / len(sorted_cluster2)
        
        ax3.plot(sorted_cluster1, cdf1, label='Cluster 1 CDF', color='blue', linewidth=2)
        ax3.plot(sorted_cluster2, cdf2, label='Cluster 2 CDF', color='orange', linewidth=2)
        
        if change_point is not None:
            ax3.axvline(change_point, color='red', linestyle='--', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
        
        ax3.set_xlabel('Index Value')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title(f'Cumulative Distribution Functions (KS={ks_statistic:.4f}, p={ks_pvalue:.4e})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sliding window
        ax4 = axes[3]
        if len(ks_scores) > 0:
            ax4.plot(positions, ks_scores, color='green', linewidth=2, label='KS Statistic (before vs after)')
            ax4.axvline(change_point, color='red', linestyle='--', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
            
            ax4.axvspan(positions[0], change_point, alpha=0.2, color='blue', label=f'Before (window={window_size})')
            ax4.axvspan(change_point, positions[-1], alpha=0.2, color='orange', label=f'After (window={window_size})')
            
            ax4.set_xlabel('Boundary Position (Index)')
            ax4.set_ylabel('KS Statistic')
            ax4.set_title(f'Sliding Window KS Test: Before vs After (window={window_size})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for change point detection', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        output_dir = Path(file1_path).parent
        output_path = output_dir / "ks_test_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_path}")
        
        plt.show()
        
        # Distribution analysis
        if change_point is not None:
            before_c1 = cluster1[cluster1 < change_point]
            before_c2 = cluster2[cluster2 < change_point]
            after_c1 = cluster1[cluster1 >= change_point]
            after_c2 = cluster2[cluster2 >= change_point]
            
            print(f"\nDistribution Analysis Around Change Point ({int(change_point)}):")
            print(f"Before: Cluster 1: {len(before_c1)} points, Cluster 2: {len(before_c2)} points")
            print(f"After:  Cluster 1: {len(after_c1)} points, Cluster 2: {len(after_c2)} points")
    
    return {
        'test_name': 'Kolmogorov-Smirnov',
        'statistic': ks_statistic,
        'pvalue': ks_pvalue,
        'change_point': int(change_point) if change_point is not None else None,
        'max_statistic': float(max_ks) if max_ks is not None else None,
        'cluster1_size': len(cluster1),
        'cluster2_size': len(cluster2)
    }


def anderson_darling_test(file1_path: str, file2_path: str, window_size: int = 100):

    cluster1 = np.load(file1_path)
    cluster2 = np.load(file2_path)
    
    print("ANDERSON-DARLING TEST")
    print(f"Cluster 1 consists of: {len(cluster1)} indices")
    print(f"Cluster 2 consists of: {len(cluster2)} indices")
    
    ad_result = anderson_ksamp([cluster1, cluster2])
    ad_statistic = ad_result.statistic
    ad_pvalue = ad_result.pvalue
    
    print(f"\nAD Statistic: {ad_statistic:.6f}")
    print(f"P-value: {ad_pvalue:.6e}")
    
    if ad_pvalue < 0.05:
        print(f"\nReject null hypothesis (p < 0.05)")
        print(f"The two clusters come from different distributions!")
    else:
        print(f"\nCannot reject null hypothesis (p >= 0.05)")
        print(f"   The two clusters come from similar/same distributions!")
    
    change_point = None
    max_ad = None
    ad_scores = []
    positions = []
    temporal_sequence = None
    time_points = None
    
    if ad_pvalue < 0.05:
        # Change Point Detection
        print(f"Change point detection (window size = {window_size}):")
        
        all_indices = np.concatenate([cluster1, cluster2])
        min_idx = int(all_indices.min())
        max_idx = int(all_indices.max())
        
        # Create temporal sequence
        temporal_sequence = []
        for idx in range(min_idx, max_idx + 1):
            in_cluster1 = idx in cluster1
            in_cluster2 = idx in cluster2
            
            if in_cluster1 and in_cluster2:
                temporal_sequence.append(2)  
            elif in_cluster1:
                temporal_sequence.append(0)  
            elif in_cluster2:
                temporal_sequence.append(1)  
            else:
                temporal_sequence.append(-1)  
        
        temporal_sequence = np.array(temporal_sequence)
        time_points = np.arange(min_idx, max_idx + 1)
        
        # Sliding window Anderson-Darling test
        ad_scores = []
        positions = []
        
        for i in range(len(temporal_sequence) - 2 * window_size):
            before_window = temporal_sequence[i:i+window_size]
            after_window = temporal_sequence[i+window_size:i+2*window_size]
            
            before_vals = before_window[before_window >= 0]
            after_vals = after_window[after_window >= 0]

            if len(before_vals) > 10 and len(after_vals) > 10:
                try:
                    ad_result = anderson_ksamp([before_vals, after_vals])
                    ad_scores.append(ad_result.statistic)
                except:
                    ad_scores.append(0)
                
                positions.append(time_points[i + window_size])  
        
        if len(ad_scores) > 0:
            ad_scores = np.array(ad_scores)
            positions = np.array(positions)
            
            change_idx = np.argmax(ad_scores)
            change_point = positions[change_idx]
            max_ad = ad_scores[change_idx]
            
            print(f"Change point detected at index: {int(change_point)}")
            print(f"Maximum AD statistic: {max_ad:.6f}")
        else:
            print("Not enough data points for change point detection")
    
    if ad_pvalue < 0.05:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Plot 1: Temporal sequence
        ax1 = axes[0]
        colors = ['blue', 'orange', 'purple', 'gray']
        
        for val, color, label in zip([0, 1, 2, -1], colors, ['Cluster 1', 'Cluster 2', 'Both', 'Neither']):
            mask = temporal_sequence == val
            if mask.any():
                ax1.scatter(time_points[mask], temporal_sequence[mask], c=color, label=label, alpha=0.6, s=10)
        
        if change_point is not None:
            ax1.axvline(change_point, color='darkred', linestyle=':', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
        
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Cluster Membership')
        ax1.set_title('Temporal Sequence of Cluster Membership')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['C1', 'C2', 'Both'])
        
        # Plot 2: Histogram
        ax2 = axes[1]
        bins = np.linspace(min(cluster1.min(), cluster2.min()), max(cluster1.max(), cluster2.max()), 50)
        ax2.hist(cluster1, bins=bins, alpha=0.6, label='Cluster 1', color='blue', density=True)
        ax2.hist(cluster2, bins=bins, alpha=0.6, label='Cluster 2', color='orange', density=True)
        
        if change_point is not None:
            ax2.axvline(change_point, color='darkred', linestyle=':', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
        
        ax2.set_xlabel('Index Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: CDFs
        ax3 = axes[2]
        sorted_cluster1 = np.sort(cluster1)
        sorted_cluster2 = np.sort(cluster2)
        cdf1 = np.arange(1, len(sorted_cluster1) + 1) / len(sorted_cluster1)
        cdf2 = np.arange(1, len(sorted_cluster2) + 1) / len(sorted_cluster2)
        
        ax3.plot(sorted_cluster1, cdf1, label='Cluster 1 CDF', color='blue', linewidth=2)
        ax3.plot(sorted_cluster2, cdf2, label='Cluster 2 CDF', color='orange', linewidth=2)
        
        if change_point is not None:
            ax3.axvline(change_point, color='darkred', linestyle=':', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
        
        ax3.set_xlabel('Index Value')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title(f'Cumulative Distribution Functions (AD={ad_statistic:.4f}, p={ad_pvalue:.4e})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sliding window
        ax4 = axes[3]
        if len(ad_scores) > 0:
            ax4.plot(positions, ad_scores, color='purple', linewidth=2, label='AD Statistic (before vs after)')
            ax4.axvline(change_point, color='darkred', linestyle=':', linewidth=2, 
                       label=f'Change Point: {int(change_point)}')
            
            ax4.axvspan(positions[0], change_point, alpha=0.2, color='blue', label=f'Before (window={window_size})')
            ax4.axvspan(change_point, positions[-1], alpha=0.2, color='orange', label=f'After (window={window_size})')
            
            ax4.set_xlabel('Boundary Position (Index)')
            ax4.set_ylabel('AD Statistic')
            ax4.set_title(f'Sliding Window Anderson-Darling Test: Before vs After (window={window_size})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for change point detection', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        output_dir = Path(file1_path).parent
        output_path = output_dir / "ad_test_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_path}")
        
        plt.show()
        
        # Distribution analysis
        if change_point is not None:
            before_c1 = cluster1[cluster1 < change_point]
            before_c2 = cluster2[cluster2 < change_point]
            after_c1 = cluster1[cluster1 >= change_point]
            after_c2 = cluster2[cluster2 >= change_point]
            
            print(f"\nDistribution Analysis Around Change Point ({int(change_point)}):")
            print(f"Before: Cluster 1: {len(before_c1)} points, Cluster 2: {len(before_c2)} points")
            print(f"After:  Cluster 1: {len(after_c1)} points, Cluster 2: {len(after_c2)} points")
    
    return {
        'test_name': 'Anderson-Darling',
        'statistic': ad_statistic,
        'pvalue': ad_pvalue,
        'change_point': int(change_point) if change_point is not None else None,
        'max_statistic': float(max_ad) if max_ad is not None else None,
        'cluster1_size': len(cluster1),
        'cluster2_size': len(cluster2)
    }


if __name__ == "__main__":
    BASE_DIR = Path("results/clustering_dbscan")
    # BASE_DIR = Path("results/clustering_kmeans")
    WINDOW_SIZE = 100

    for exp_dir in sorted(BASE_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        file1 = exp_dir / "sklearn_cluster0.npy"
        file2 = exp_dir / "sklearn_cluster1.npy"

        if not file1.exists() or not file2.exists():
            print(f"Skipping {exp_dir.name} (missing cluster files)")
            continue

        log_file = exp_dir / "ks_ad_analysis.log"

        with open(log_file, "w") as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print("=" * 80)
            print(f"EXPERIMENT: {exp_dir.name}")
            print("=" * 80)

            ks_results = kolmogorov_smirnov_test(
                str(file1), str(file2), window_size=WINDOW_SIZE
            )

            ad_results = anderson_darling_test(
                str(file1), str(file2), window_size=WINDOW_SIZE
            )

            print("\nCOMPARISON SUMMARY")
            print(f"KS Change Point: {ks_results['change_point']}")
            print(f"AD Change Point: {ad_results['change_point']}")

            sys.stdout = original_stdout
# if __name__ == "__main__":
#     WINDOW_SIZE = 100

#     file1 = Path("/Users/ashwin/Desktop/AMD_simulated_benchmarks/results/clustering_dbscan/bm_n6000_GenBIN_Fr0.950_mpri15.300_spri0.400_msec1.000_ssec1.330_WarmA0.000_c0_w_0/amd_cluster0.npy")
#     file2 = Path("/Users/ashwin/Desktop/AMD_simulated_benchmarks/results/clustering_dbscan/bm_n6000_GenBIN_Fr0.950_mpri15.300_spri0.400_msec1.000_ssec1.330_WarmA0.000_c0_w_0/amd_cluster1.npy")

#     if not file1.exists() or not file2.exists():
#         print("One or both input files do not exist")
#         sys.exit(1)

#     log_file = file1.parent / "ks_ad_analysis.log"

#     with open(log_file, "w") as f:
#         original_stdout = sys.stdout
#         sys.stdout = f

#         print("=" * 80)
#         print("MANUAL EXPERIMENT RUN")
#         print("=" * 80)
#         print(f"Cluster 1 file: {file1}")
#         print(f"Cluster 2 file: {file2}")

#         ks_results = kolmogorov_smirnov_test(
#             str(file1), str(file2), window_size=WINDOW_SIZE
#         )

#         ad_results = anderson_darling_test(
#             str(file1), str(file2), window_size=WINDOW_SIZE
#         )

#         print("\nCOMPARISON SUMMARY")
#         print(f"KS Change Point: {ks_results['change_point']}")
#         print(f"AD Change Point: {ad_results['change_point']}")

#         sys.stdout = original_stdout
