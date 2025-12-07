"""
Sampling methods for genetic sequence evolution.
"""
import torch
from typing import Dict, List, Tuple


def metropolis_step_batch(
    chains: torch.Tensor,
    dna_chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    codon_to_amino: Dict[str, int],
    all_codons: List[str],
    device: torch.device,
    dtype: torch.dtype,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform batched Metropolis sampling step for gap insertion/deletion.
    
    For each chain:
    - Randomly select one position
    - If position is a gap (---): propose random non-gap codon uniformly
    - If position is non-gap: propose gap (---) with probability 1/64
    - Accept/reject based on amino acid level energetics
    
    Args:
        chains: One-hot encoded amino acid sequences (N, L, q)
        dna_chains: DNA sequences as codon indices (N, L)
        params: DCA model parameters with "bias" (L, q) and "coupling_matrix" (L, q, L, q)
        codon_to_amino: Mapping from codon string to amino acid index
        all_codons: List of all codons (length 62: 61 standard + 1 gap)
        device: Torch device (CPU/GPU)
        dtype: Torch data type
        beta: Inverse temperature (default: 1.0)
    
    Returns:
        Tuple of (updated amino acid chains, updated DNA chains)
    """
    N, L, q = chains.shape
    
    # Gap codon is '---' at index 0 in amino acids
    gap_codon_str = '---'
    gap_aa_idx = 0  # Gap amino acid is always at index 0
    
    # Find gap codon index in all_codons list
    gap_codon_idx = all_codons.index(gap_codon_str)
    
    # Build tensor mapping codon index -> amino acid index
    codon_to_aa_tensor = torch.zeros(len(all_codons), dtype=torch.long, device=device)
    for codon_idx, codon_str in enumerate(all_codons):
        codon_to_aa_tensor[codon_idx] = codon_to_amino[codon_str]
    
    # Non-gap codons: all except gap
    non_gap_codon_indices = [i for i, codon in enumerate(all_codons) if codon != gap_codon_str]
    num_non_gap_codons = len(non_gap_codon_indices)  # Should be 61
    non_gap_codon_tensor = torch.tensor(non_gap_codon_indices, dtype=torch.long, device=device)
    
    # Randomly select one position for each chain
    batch_arange = torch.arange(N, device=device)
    selected_sites = torch.randint(0, L, (N,), device=device)
    
    # Get current codons and amino acids at selected sites
    current_codon_indices = dna_chains[batch_arange, selected_sites]  # Shape: (N,)
    current_aa_onehot = chains[batch_arange, selected_sites]  # Shape: (N, q)
    current_aa_indices = current_aa_onehot.argmax(dim=-1)  # Shape: (N,)
    
    # Determine which chains have gap at selected position
    is_gap_mask = (current_codon_indices == gap_codon_idx)  # Shape: (N,)
    
    # Propose new codons
    proposed_codon_indices = torch.zeros(N, dtype=torch.long, device=device)
    
    # For gap positions: propose random non-gap codon uniformly
    num_gaps = is_gap_mask.sum().item()
    if num_gaps > 0:
        random_non_gap_idx = torch.randint(0, num_non_gap_codons, (num_gaps,), device=device)
        proposed_codon_indices[is_gap_mask] = non_gap_codon_tensor[random_non_gap_idx]
    
    # For non-gap positions: propose gap
    num_non_gaps = (~is_gap_mask).sum().item()
    if num_non_gaps > 0:
        proposed_codon_indices[~is_gap_mask] = gap_codon_idx
    
    # Convert proposed codons to amino acid indices using pre-built tensor
    proposed_aa_indices = codon_to_aa_tensor[proposed_codon_indices]  # Shape: (N,)
    
    # Create proposed amino acid one-hot
    proposed_aa_onehot = torch.nn.functional.one_hot(proposed_aa_indices, num_classes=q).to(dtype)
    
    # Compute energies using Metropolis criterion
    # Extract biases and couplings for selected sites
    biases = params["bias"][selected_sites]  # Shape: (N, q)
    couplings_batch = params["coupling_matrix"][selected_sites]  # Shape: (N, q, L, q)
    
    # Compute coupling term
    chains_flat = chains.reshape(N, L * q, 1)
    couplings_flat = couplings_batch.reshape(N, q, L * q)
    coupling_term = torch.bmm(couplings_flat, chains_flat).squeeze(-1)  # (N, q)
    
    # Local field
    local_field = biases + coupling_term  # Shape: (N, q)
    
    # Energy difference: ΔE = (old - new) · local_field
    delta_E = torch.sum((current_aa_onehot - proposed_aa_onehot) * local_field, dim=-1)  # Shape: (N,)
    
    # Metropolis acceptance: P_accept = exp(-beta * ΔE)
    acceptance_prob = torch.exp(-beta * delta_E)
    
    # For non-gap -> gap transitions, multiply by proposal probability 1/64
    acceptance_prob[~is_gap_mask] = acceptance_prob[~is_gap_mask] / 64.0
    
    # Clip acceptance probability to [0, 1]
    acceptance_prob = torch.clamp(acceptance_prob, 0.0, 1.0)
    
    # Accept/reject
    random_uniform = torch.rand(N, device=device, dtype=dtype)
    accept_mask = (random_uniform < acceptance_prob)  # Shape: (N,)
    
    # Update chains
    final_aa_onehot = torch.where(accept_mask.unsqueeze(-1), proposed_aa_onehot, current_aa_onehot)
    final_codon_indices = torch.where(accept_mask, proposed_codon_indices, current_codon_indices)
    
    updated_chains = chains.clone()
    updated_chains[batch_arange, selected_sites] = final_aa_onehot
    
    updated_dna_chains = dna_chains.clone()
    updated_dna_chains[batch_arange, selected_sites] = final_codon_indices
    
    return updated_chains, updated_dna_chains


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
