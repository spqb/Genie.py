"""
Sampling module for genetic sequence evolution using Gibbs sampling.
This module provides GPU-optimized sampling functions for amino acid sequences.
"""
from typing import Dict, Callable

import torch
from torch.nn.functional import one_hot


# ============================================================================
# PROFILE SAMPLING - Initialize sequences from independent site frequencies
# ============================================================================

def sampling_profile(
    params: Dict[str, torch.Tensor],
    nsamples: int,
    beta: float,
) -> torch.Tensor:
    """
    Sample from the profile model defined by local biases only (no couplings).
    
    This function generates sequences by sampling independently at each position
    according to the local bias (single-site frequencies). No correlations are
    considered, making it suitable for initialization.
    
    Args:
        params: Dictionary containing DCA model parameters
            - "bias": Tensor of shape (L, q) containing local biases
        nsamples: Number of sequences to generate
        beta: Inverse temperature for sampling (higher = more deterministic)
        
    Returns:
        One-hot encoded sequences of shape (nsamples, L, q)
    """
    L, q = params["bias"].shape
    device = params["bias"].device
    dtype = params["bias"].dtype
    
    # Scale biases by inverse temperature and expand for batch processing
    logits = beta * params["bias"].unsqueeze(0).expand(nsamples, -1, -1)  # Shape: (nsamples, L, q)
    
    # Sample indices from categorical distribution
    sampled_indices = torch.multinomial(
        torch.softmax(logits.view(-1, q), dim=-1), 
        num_samples=1
    ).squeeze(-1)
    
    # Convert to one-hot encoding
    sampled_sequences = one_hot(sampled_indices, num_classes=q).view(nsamples, L, q).to(dtype).to(device)
    
    return sampled_sequences


# ============================================================================
# GIBBS SAMPLING - Single mutation step with full DCA model
# ============================================================================

def gibbs_step_independent_sites(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a single Gibbs sampling mutation step with independent site selection.
    
    This function selects a different random site for each sequence in the batch
    and updates it using Gibbs sampling. This approach is more suitable when starting
    from the same wild-type sequence, as mutations are independent across chains.
    
    Algorithm:
    1. Select random site for each sequence (different sites per sequence)
    2. Compute local field (bias + coupling term) for selected sites
    3. Sample new amino acid from Boltzmann distribution
    4. Update sequences in-place
    
    Args:
        chains: One-hot encoded sequences of shape (N, L, q)
        params: Dictionary containing DCA model parameters
            - "bias": Tensor of shape (L, q) containing local biases
            - "coupling_matrix": Tensor of shape (L, q, L, q) containing pairwise couplings
        beta: Inverse temperature (default: 1.0)

    Returns:
        Tuple of (updated_chains, mutation_info) where:
            - updated_chains: Modified sequences with mutations applied
            - mutation_info: Tensor (N, 2) containing [site_index, new_aa_index] for each chain
    """
    N, L, q = chains.shape
    device = chains.device
    dtype = chains.dtype
    
    # Select different random site for each sequence
    idx_batch = torch.randint(0, L, (N,), device=device)  # Shape: (N,)
    
    # Extract biases and couplings for selected sites
    biases = params["bias"][idx_batch]  # Shape: (N, q)
    couplings_batch = params["coupling_matrix"][idx_batch]  # Shape: (N, q, L, q)
    
    # Compute coupling term using optimized batch matrix multiplication
    chains_flat = chains.reshape(N, L * q, 1)  # Shape: (N, L*q, 1)
    couplings_flat = couplings_batch.reshape(N, q, L * q)  # Shape: (N, q, L*q)
    coupling_term = torch.bmm(couplings_flat, chains_flat).squeeze(-1)  # Shape: (N, q)
    
    # Compute local field (total energy contribution for each amino acid)
    logits = beta * (biases + coupling_term)  # Shape: (N, q)
    
    # Sample new amino acids from Boltzmann distribution
    new_residues = one_hot(
        torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1), 
        num_classes=q
    ).to(dtype)  # Shape: (N, q)
    
    # Update sequences in-place using advanced indexing
    batch_arange = torch.arange(N, device=device)
    chains[batch_arange, idx_batch] = new_residues

    # Prepare mutation info tensor for tracking (optional analysis)
    selected_sites_tensor = idx_batch.unsqueeze(-1)  # Shape: (N, 1)
    new_aa_tensor = new_residues.argmax(dim=-1)  # Shape: (N,)
    mutation_info_tensor = torch.cat(
        (selected_sites_tensor, new_aa_tensor.unsqueeze(-1)), 
        dim=-1
    )  # Shape: (N, 2)

    return chains, mutation_info_tensor


# ============================================================================
# GIBBS SAMPLING - Main interface function
# ============================================================================

def gibbs_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Gibbs sampling with a single mutation attempt per sequence.
    
    This is the main interface for Gibbs sampling used in the evolution loop.
    It creates a copy of the input to avoid modifying it in-place, then applies
    one mutation step using gibbs_step_independent_sites.
    
    Args:
        chains: Initial one-hot encoded sequences of shape (N, L, q)
        params: Dictionary containing DCA model parameters
            - "bias": Tensor of shape (L, q) containing local biases
            - "coupling_matrix": Tensor of shape (L, q, L, q) containing pairwise couplings
        beta: Inverse temperature (default: 1.0)
    
    Returns:
        Tuple of (updated_chains, mutation_info) where:
            - updated_chains: Modified sequences with mutations applied
            - mutation_info: Tensor (N, 2) containing [site_index, new_aa_index] for each chain
    """
    # Clone chains to avoid modifying original
    chains_mutate = chains.clone()
    
    # Perform single Gibbs step
    chains_mutate, mutation_info_tensor = gibbs_step_independent_sites(chains_mutate, params, beta)

    return chains_mutate, mutation_info_tensor


# ============================================================================
# SAMPLER FACTORY - Get sampling function by name
# ============================================================================

def get_sampler(sampling_method: str) -> Callable:
    """
    Return the sampling function corresponding to the chosen method.
    
    This factory function allows dynamic selection of sampling algorithms
    based on string input, facilitating easy switching between methods.

    Args:
        sampling_method: String indicating the sampling method
            - "gibbs": Single Gibbs step per call
            - "gibbs_sweep": Multiple Gibbs steps per call (not currently implemented)

    Returns:
        Callable sampling function

    Raises:
        KeyError: If sampling_method is not recognized
    """
    if sampling_method == "gibbs":
        return gibbs_sampling
    elif sampling_method == "gibbs_sweep":
        # Note: gibbs_sweep not currently used in main workflow
        return gibbs_sampling_sweep
    else:
        raise KeyError(f"Unknown sampling method '{sampling_method}'. Choose 'gibbs' or 'gibbs_sweep'.")

