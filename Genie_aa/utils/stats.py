"""
Statistical analysis utilities for sequence evolution.
Provides functions to compute and track correlations between target and evolved sequences.
"""
import torch
import numpy as np
from adabmDCA import get_freq_single_point, get_freq_two_points, get_correlation_two_points


def compute_correlations(
    pi_target: np.ndarray,
    pij_target: np.ndarray,
    chains: torch.Tensor
) -> float:
    """
    Compute Pearson correlation between target and current sequence statistics.
    
    This function measures how well the current evolved sequences match the target
    distribution by comparing both single-site and pairwise frequencies.
    
    Args:
        pi_target: Target single-point frequencies of shape (L, q)
                  where L is sequence length and q is alphabet size
        pij_target: Target two-point frequencies of shape (L, L, q, q)
                   containing all pairwise amino acid correlations
        chains: Current one-hot encoded sequences of shape (N, L, q)
               where N is the number of sequences
    
    Returns:
        float: Pearson correlation coefficient between target and current
              two-point statistics (ranges from -1 to 1, where 1 is perfect match)
    """
    # Compute current frequencies from evolved sequences
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    
    # Compute Pearson correlation between target and current statistics
    pearson, slope = get_correlation_two_points(
        fij=pij_target,
        pij=pij,
        fi=pi_target,
        pi=pi
    )
    
    return pearson


def track_evolution_statistics(
    pi_target: np.ndarray,
    pij_target: np.ndarray,
    chains_history: list,
    iteration_interval: int = 100
) -> dict:
    """
    Track statistical evolution over iterations.
    
    This function computes Pearson correlations for a sequence of checkpointed
    chains to monitor convergence during evolution.
    
    Args:
        pi_target: Target single-point frequencies of shape (L, q)
        pij_target: Target two-point frequencies of shape (L, L, q, q)
        chains_history: List of chains tensors at different iteration checkpoints
        iteration_interval: Number of iterations between each saved checkpoint
    
    Returns:
        dict: Dictionary containing:
            - 'iterations': List of iteration numbers
            - 'pearson': List of Pearson correlation values at each checkpoint
    """
    iterations = []
    pearson_values = []
    
    # Compute correlation for each checkpoint
    for i, chains in enumerate(chains_history):
        iteration = i * iteration_interval
        pearson = compute_correlations(pi_target, pij_target, chains)
        
        iterations.append(iteration)
        pearson_values.append(pearson)
    
    return {
        'iterations': iterations,
        'pearson': pearson_values
    }


def compute_target_statistics(
    sequences: torch.Tensor, 
    weights: torch.Tensor = None, 
    pseudo_count: float = 0
) -> tuple:
    """
    Compute target statistics from reference sequences.
    
    This function extracts single-site and pairwise frequencies from a reference
    dataset to use as evolutionary targets.
    
    Args:
        sequences: One-hot encoded reference sequences of shape (N, L, q)
        weights: Optional sequence weights of shape (N,) for reweighting
                (default: None, uses uniform weights)
        pseudo_count: Pseudo-count for frequency regularization
                     (default: 0, no regularization)
    
    Returns:
        tuple: (pi_target, pij_target) where:
            - pi_target: Single-site frequencies of shape (L, q)
            - pij_target: Pairwise frequencies of shape (L, L, q, q)
    """
    # Compute single-site frequencies
    pi_target = get_freq_single_point(
        data=sequences, 
        weights=weights, 
        pseudo_count=pseudo_count
    )
    
    # Compute pairwise frequencies
    pij_target = get_freq_two_points(
        data=sequences, 
        weights=weights, 
        pseudo_count=pseudo_count
    )
    
    return pi_target, pij_target

