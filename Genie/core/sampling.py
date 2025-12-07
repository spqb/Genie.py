"""
Sampling methods for genetic sequence evolution.
"""
import torch
from typing import Dict, List, Tuple


def metropolis_step_batch(
    chains: torch.Tensor,
    params: Dict,
    codon_to_amino: Dict[str, int],
    amino_to_codons: Dict[int, List[str]],
    codon_neighbors: Dict[str, Dict[int, List[str]]],
    neighbor_counts: Dict[Tuple[str, int], int],
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Perform batched Metropolis sampling step on multiple sequences.
    
    Optimized for parallel GPU execution.
    
    Args:
        chains: Batch of sequences to evolve (batch_size, seq_length)
        params: DCA model parameters
        codon_to_amino: Dictionary mapping codons to amino acid indices
        amino_to_codons: Dictionary mapping amino acid indices to codon lists
        codon_neighbors: Dictionary of codon nearest neighbors
        neighbor_counts: Dictionary of neighbor counts per codon position
        device: Torch device (CPU/GPU)
        dtype: Torch data type
    
    Returns:
        Tensor: Evolved sequences (batch_size, seq_length)
    """
    # TODO: Implement batched Metropolis sampling logic
    # Key optimization: vectorize all operations across batch dimension
    # Placeholder - return chains unchanged for now
    return chains


def gibbs_step_batch(
    chains: torch.Tensor, 
    dna_chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    codon_neighbor_tensor: torch.Tensor,
    mutation_lookup: torch.Tensor,
    num_options: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform batched Gibbs sampling step on multiple sequences.
    
    Fully GPU-optimized - all operations vectorized on GPU.
    
    Args:
        chains: One-hot encoded amino acid sequences (batch_size, seq_length, q)
        dna_chains: DNA sequences as codon indices (batch_size, seq_length) - integer tensor
        params: DCA model parameters with keys:
            - "bias": Local biases (seq_length, q)
            - "coupling_matrix": Coupling matrix (seq_length, q, seq_length, q)
        codon_neighbor_tensor: Pre-computed neighbor accessibility (num_codons, 3, q)
        mutation_lookup: Pre-computed codon mutations (num_codons, 3, q, max_neighbors)
        num_options: Count of valid options (num_codons, 3, q)
        device: Torch device (CPU/GPU)
        dtype: Torch data type
        beta: Inverse temperature (default: 1.0)
    
    Returns:
        Tuple of (updated amino acid chains, updated DNA chains)
    """
    N, L, q = chains.shape
    
    # Identify gap positions - chains[:, :, 0] is gap channel in one-hot
    gap_mask = chains[:, :, 0] == 1.0  # Shape: (N, L)
    valid_sites_mask = ~gap_mask
    
    # Select one random valid site for each chain (vectorized)
    num_valid_sites = valid_sites_mask.sum(dim=1)  # Shape: (N,)
    rand_floats = torch.rand(N, device=device)
    rand_valid_idx = (rand_floats * num_valid_sites.float()).long()
    # Clamp element-wise: max with 0, min with num_valid_sites-1
    rand_valid_idx = torch.maximum(rand_valid_idx, torch.zeros_like(rand_valid_idx))
    rand_valid_idx = torch.minimum(rand_valid_idx, num_valid_sites - 1)
    
    cumsum_mask = torch.cumsum(valid_sites_mask.long(), dim=1)
    select_mask = cumsum_mask == (rand_valid_idx.unsqueeze(1) + 1)
    selected_sites = torch.argmax(select_mask.long(), dim=1)  # Shape: (N,)
    
    # Randomly select nucleotide position (0, 1, or 2) for each chain
    nucleotide_positions = torch.randint(0, 3, (N,), device=device)
    
    # Extract biases and couplings for selected sites
    batch_arange = torch.arange(N, device=device)
    biases = params["bias"][selected_sites]  # Shape: (N, q)
    couplings_batch = params["coupling_matrix"][selected_sites]  # Shape: (N, q, L, q)
    
    # Compute coupling term
    chains_flat = chains.reshape(N, L * q, 1)
    couplings_flat = couplings_batch.reshape(N, q, L * q)
    coupling_term = torch.bmm(couplings_flat, chains_flat).squeeze(-1)  # (N, q)
    
    # Compute logits for all amino acids
    logits_full = beta * (biases + coupling_term)  # Shape: (N, q)
    
    # Build valid amino acid mask (FULLY VECTORIZED)
    current_codon_indices = dna_chains[batch_arange, selected_sites]  # Shape: (N,)
    valid_aa_mask = codon_neighbor_tensor[current_codon_indices, nucleotide_positions]  # (N, q)
    
    # Mask invalid amino acids with -inf
    logits_masked = logits_full.clone()
    logits_masked[~valid_aa_mask] = float('-inf')
    
    # Sample new amino acids from masked distribution
    probs = torch.softmax(logits_masked, dim=-1)
    new_aa_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Shape: (N,)
    
    # Create new one-hot residues
    new_residues = torch.nn.functional.one_hot(new_aa_indices, num_classes=q).to(dtype)
    
    # Update amino acid chains
    updated_chains = chains.clone()
    updated_chains[batch_arange, selected_sites] = new_residues
    
    # Update DNA chains (FULLY VECTORIZED)
    # The new codon is determined by: current_codon + mutation at nucleotide_position -> new_aa
    # mutation_lookup[current_codon, nuc_pos, new_aa] contains the specific codon(s)
    # that result from mutating nucleotide at position nuc_pos to yield new_aa
    
    # Get the valid codon options for each chain
    # Note: there's typically only 1 option (the specific mutation at that position)
    # but could be multiple if different nucleotides at that position code for same aa
    valid_codons = mutation_lookup[current_codon_indices, nucleotide_positions, new_aa_indices]  # (N, max_neighbors)
    
    # Get number of options (typically 1, but could be more)
    counts = num_options[current_codon_indices, nucleotide_positions, new_aa_indices]  # Shape: (N,)
    
    # Generate random selection within valid range for each chain (in case multiple options exist)
    rand_selection = torch.floor(torch.rand(N, device=device) * counts.float()).long()
    # Clamp element-wise
    rand_selection = torch.maximum(rand_selection, torch.zeros_like(rand_selection))
    rand_selection = torch.minimum(rand_selection, counts - 1)
    
    # Select the new codon for each chain
    new_codon_indices = valid_codons[batch_arange, rand_selection]  # Shape: (N,)
    
    # Update DNA chains
    updated_dna_chains = dna_chains.clone()
    updated_dna_chains[batch_arange, selected_sites] = new_codon_indices
    
    return updated_chains, updated_dna_chains
