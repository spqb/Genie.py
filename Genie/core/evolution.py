"""
Evolution module for genetic sequence evolution using DCA models.
"""
import torch
from typing import Dict, List, Tuple
from .sampling import metropolis_step_batch, gibbs_step_batch


def evolve_sequences(
    chains: torch.Tensor,
    dna_chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    codon_neighbor_tensor: torch.Tensor,
    codon_neighbor_codon_tensor: torch.Tensor,
    mutation_lookup: torch.Tensor,
    num_options: torch.Tensor,
    codon_usage: torch.Tensor,
    p: float = 0.5,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evolve a set of sequences using Metropolis and Gibbs sampling.
    
    Optimized for GPU execution with millions of iterations.
    
    Args:
        chains: One-hot encoded amino acid sequences (n_chains, seq_length, q)
        dna_chains: DNA sequences as codon indices (n_chains, seq_length)
        params: DCA model parameters with bias and coupling_matrix
        codon_neighbor_tensor: Pre-computed neighbor accessibility (num_codons, 3, q)
        codon_neighbor_codon_tensor: Pre-computed codon neighbor accessibility (num_codons, 3, num_codons)
        mutation_lookup: Pre-computed codon mutations (num_codons, 3, q, max_neighbors)
        num_options: Count of valid options (num_codons, 3, q)
        codon_usage: Tensor (num_codons,) with codon usage frequencies
        p: Float probability threshold for Metropolis vs Gibbs selection
        device: Torch device (CPU/GPU)
        dtype: Torch data type
        beta: Inverse temperature for Gibbs sampling
    
    Returns:
        Tuple of (evolved amino acid chains, evolved DNA chains)
    """
    n_chains = chains.shape[0]
    
    # Generate random numbers on GPU for each chain
    random_values = torch.rand(n_chains, device=device)
    
    # Create boolean mask for Metropolis vs Gibbs
    gibbs_mask = random_values > p
    
    # Get indices for each method
    gibbs_indices = torch.where(gibbs_mask)[0]
    metropolis_indices = torch.where(~gibbs_mask)[0]
    
    # Clone chains to avoid in-place modifications
    evolved_chains = chains.clone()
    evolved_dna_chains = dna_chains.clone()
    
    # Process Metropolis chains in batch if any exist
    if metropolis_indices.numel() > 0:
        metropolis_chains = chains[metropolis_indices]
        metropolis_dna_chains = dna_chains[metropolis_indices]
        
        evolved_metropolis, evolved_metropolis_dna = metropolis_step_batch(
            chains=metropolis_chains,
            dna_chains=metropolis_dna_chains,
            params=params,
            codon_to_amino=params["codon_to_amino"],
            all_codons=params["all_codons"],
            device=device,
            dtype=dtype,
            beta=beta
        )
        
        evolved_chains[metropolis_indices] = evolved_metropolis
        evolved_dna_chains[metropolis_indices] = evolved_metropolis_dna
    
    # Process Gibbs chains in batch if any exist
    if gibbs_indices.numel() > 0:
        gibbs_chains = chains[gibbs_indices]
        gibbs_dna_chains = dna_chains[gibbs_indices]
        
        evolved_gibbs, evolved_gibbs_dna = gibbs_step_batch(
            chains=gibbs_chains,
            dna_chains=gibbs_dna_chains,
            params=params,
            codon_neighbor_tensor=codon_neighbor_tensor,
            codon_neighbor_codon_tensor=codon_neighbor_codon_tensor,
            mutation_lookup=mutation_lookup,
            num_options=num_options,
            codon_usage=codon_usage,
            codon_to_aa_idx=params["codon_to_aa_idx"],
            device=device,
            dtype=dtype,
            beta=beta
        )
        
        evolved_chains[gibbs_indices] = evolved_gibbs
        evolved_dna_chains[gibbs_indices] = evolved_gibbs_dna
    
    return evolved_chains, evolved_dna_chains

