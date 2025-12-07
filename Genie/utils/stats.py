"""
Statistical analysis utilities for sequence evolution.
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
    
    Args:
        pi_target: Target single-point frequencies (L, q)
        pij_target: Target two-point frequencies (L, L, q, q)
        chains: Current one-hot encoded sequences (N, L, q)
    
    Returns:
        float: Pearson correlation coefficient
    """
    # Compute current frequencies
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    
    # Compute correlation
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
    
    Args:
        pi_target: Target single-point frequencies (L, q)
        pij_target: Target two-point frequencies (L, L, q, q)
        chains_history: List of chains at different iterations
        iteration_interval: Interval at which chains were saved
    
    Returns:
        dict: Dictionary with iterations and pearson correlations
    """
    iterations = []
    pearson_values = []
    
    for i, chains in enumerate(chains_history):
        iteration = i * iteration_interval
        pearson = compute_correlations(pi_target, pij_target, chains)
        
        iterations.append(iteration)
        pearson_values.append(pearson)
    
    return {
        'iterations': iterations,
        'pearson': pearson_values
    }


def compute_target_statistics(sequences: torch.Tensor) -> tuple:
    """
    Compute target statistics from reference sequences.
    
    Args:
        sequences: One-hot encoded reference sequences (N, L, q)
    
    Returns:
        tuple: (pi_target, pij_target)
    """
    pi_target = get_freq_single_point(data=sequences)
    pij_target = get_freq_two_points(data=sequences)
    
    return pi_target, pij_target
