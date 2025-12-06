"""
Evolution module for genetic sequence evolution using DCA models.
"""
import torch
from typing import Dict, List, Tuple
from .sampling import metropolis_step_batch, gibbs_step_batch


def evolve_sequences(
    chains: torch.Tensor,
    params: Dict,
    codon_to_amino: Dict[str, int],
    amino_to_codons: Dict[int, List[str]],
    codon_neighbors: Dict[str, Dict[int, List[str]]],
    neighbor_counts: Dict[Tuple[str, int], int],
    p: float = 0.5,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Evolve a set of sequences using Metropolis and Gibbs sampling.
    
    Optimized for GPU execution with millions of iterations.
    
    Args:
        chains: Tensor of sequences to evolve (n_chains, seq_length)
        params: DCA model parameters
        codon_to_amino: Dictionary mapping codons to amino acid indices
        amino_to_codons: Dictionary mapping amino acid indices to codon lists
        codon_neighbors: Dictionary of codon nearest neighbors
        neighbor_counts: Dictionary of neighbor counts per codon position
        p: Float probability threshold for Metropolis vs Gibbs selection
        device: Torch device (CPU/GPU)
        dtype: Torch data type
    
    Returns:
        Tensor: Evolved sequences in the same order as input
    """
    n_chains = chains.shape[0]
    
    # Generate random numbers on GPU for each chain
    random_values = torch.rand(n_chains, device=device)
    
    # Create boolean mask for Metropolis vs Gibbs
    metropolis_mask = random_values < p
    
    # Get indices for each method
    metropolis_indices = torch.where(metropolis_mask)[0]
    gibbs_indices = torch.where(~metropolis_mask)[0]
    
    # Clone chains to avoid in-place modifications
    evolved_chains = chains.clone()
    
    # Process Metropolis chains in batch if any exist
    if metropolis_indices.numel() > 0:
        metropolis_chains = chains[metropolis_indices]
        evolved_metropolis = metropolis_step_batch(
            metropolis_chains, params, codon_to_amino, amino_to_codons,
            codon_neighbors, neighbor_counts, device, dtype
        )
        evolved_chains[metropolis_indices] = evolved_metropolis
    
    # Process Gibbs chains in batch if any exist
    if gibbs_indices.numel() > 0:
        gibbs_chains = chains[gibbs_indices]
        evolved_gibbs = gibbs_step_batch(
            gibbs_chains, params, codon_to_amino, amino_to_codons,
            codon_neighbors, neighbor_counts, device, dtype
        )
        evolved_chains[gibbs_indices] = evolved_gibbs
    
    return evolved_chains

