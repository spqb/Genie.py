"""
PCA utilities for sequence analysis.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def train_pca(sequences_onehot: torch.Tensor, n_components: int = 2, weights: torch.Tensor = None, max_samples: int = 5000):
    """
    Train PCA on one-hot encoded sequences with optional sequence weighting.
    
    If N > max_samples, subsample max_samples sequences randomly for PCA training.
    
    Args:
        sequences_onehot: One-hot encoded sequences (N, L, q)
        n_components: Number of principal components
        weights: Sequence weights (N,) for reweighting, optional
        max_samples: Maximum number of samples to use for PCA training (default: 5000)
    
    Returns:
        Tuple: (pca, subsample_indices)
            - pca: Fitted PCA object
            - subsample_indices: Indices of subsampled sequences (None if no subsampling)
    """
    N, L, q = sequences_onehot.shape
    
    # Subsample if N > max_samples
    subsample_indices = None
    if N > max_samples:
        print(f"  Subsampling {max_samples} sequences from {N} for PCA training...")
        # Random sampling
        subsample_indices = np.random.choice(N, size=max_samples, replace=False)
        subsample_indices.sort()  # Sort for reproducibility
        sequences_onehot = sequences_onehot[subsample_indices]
        if weights is not None:
            weights = weights[subsample_indices]
        N = max_samples
    
    # Flatten sequences to (N, L*q)
    sequences_flat = sequences_onehot.reshape(N, L * q).cpu().numpy()
    
    # Convert weights to numpy if provided
    if weights is not None:
        weights_np = weights.cpu().numpy() if isinstance(weights, torch.Tensor) else weights
        # Normalize weights to sum to N (preserve sample size)
        weights_np = weights_np * (N / weights_np.sum())
    else:
        weights_np = None
    
    # Train PCA with weighted samples
    pca = PCA(n_components=n_components)
    
    if weights_np is not None:
        # Weighted PCA: center data with weights, then compute covariance
        mean_weighted = np.average(sequences_flat, axis=0, weights=weights_np)
        sequences_centered = sequences_flat - mean_weighted
        
        # Weight the centered data
        sequences_weighted = sequences_centered * np.sqrt(weights_np)[:, np.newaxis]
        
        # Fit PCA on weighted data
        pca.fit(sequences_weighted)
        
        # Adjust mean to use weighted mean
        pca.mean_ = mean_weighted
    else:
        # Standard PCA without weights
        pca.fit(sequences_flat)
    
    return pca, subsample_indices


def project_sequences(sequences_onehot: torch.Tensor, pca):
    """
    Project sequences onto PCA space.
    
    Args:
        sequences_onehot: One-hot encoded sequences (N, L, q)
        pca: Fitted PCA object
    
    Returns:
        Projected coordinates (N, n_components)
    """
    N, L, q = sequences_onehot.shape
    
    # Flatten sequences to (N, L*q)
    sequences_flat = sequences_onehot.reshape(N, L * q).cpu().numpy()
    
    # Project
    projected = pca.transform(sequences_flat)
    
    return projected


def plot_pca_evolution(
    initial_proj: np.ndarray,
    final_proj: np.ndarray,
    output_path: str,
    title: str = "PCA: Sequence Evolution",
    natural_proj: np.ndarray = None
):
    """
    Plot PCA projection of sequences before and after evolution.
    
    Args:
        initial_proj: Initial sequence projections (N, 2)
        final_proj: Final sequence projections (N, 2)
        output_path: Path to save the plot
        title: Plot title
        natural_proj: Natural sequences projections (N, 2), optional
    """
    plt.figure(figsize=(10, 8))
    
    # Plot natural sequences first (background)
    if natural_proj is not None:
        plt.scatter(
            natural_proj[:, 0], 
            natural_proj[:, 1],
            c='gray', 
            alpha=0.2, 
            s=10,
            label='Natural sequences',
            zorder=1
        )
    
    # Plot initial sequences
    plt.scatter(
        initial_proj[:, 0], 
        initial_proj[:, 1],
        c='blue', 
        alpha=0.5, 
        s=20,
        label='Initial sequences',
        zorder=2
    )
    
    # Plot final sequences
    plt.scatter(
        final_proj[:, 0], 
        final_proj[:, 1],
        c='red', 
        alpha=0.5, 
        s=20,
        label='Final sequences (after evolution)',
        zorder=3
    )
    
    # Plot trajectories for a subset of sequences
    num_trajectories = min(100, len(initial_proj))
    for i in range(num_trajectories):
        plt.plot(
            [initial_proj[i, 0], final_proj[i, 0]],
            [initial_proj[i, 1], final_proj[i, 1]],
            'gray',
            alpha=0.1,
            linewidth=0.5
        )
    
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA plot saved to {output_path}")


def plot_pca_with_density(
    initial_proj: np.ndarray,
    final_proj: np.ndarray,
    output_path: str,
    title: str = "PCA: Sequence Evolution with Density",
    natural_proj: np.ndarray = None
):
    """
    Plot PCA projection with density contours.
    
    Args:
        initial_proj: Initial sequence projections (N, 2)
        final_proj: Final sequence projections (N, 2)
        output_path: Path to save the plot
        title: Plot title
        natural_proj: Natural sequences projections (N, 2), optional
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot initial
    if natural_proj is not None:
        ax1.scatter(
            natural_proj[:, 0], 
            natural_proj[:, 1],
            c='gray', 
            alpha=0.2, 
            s=5,
            label='Natural sequences'
        )
    ax1.scatter(
        initial_proj[:, 0], 
        initial_proj[:, 1],
        c='blue', 
        alpha=0.3, 
        s=10,
        label='Initial sequences'
    )
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('Initial Sequences', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot final
    if natural_proj is not None:
        ax2.scatter(
            natural_proj[:, 0], 
            natural_proj[:, 1],
            c='gray', 
            alpha=0.2, 
            s=5,
            label='Natural sequences'
        )
    ax2.scatter(
        final_proj[:, 0], 
        final_proj[:, 1],
        c='red', 
        alpha=0.3, 
        s=10,
        label='Final sequences'
    )
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_title('Final Sequences (after evolution)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA density plot saved to {output_path}")
