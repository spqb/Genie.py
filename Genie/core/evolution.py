"""
Evolution module for genetic sequence evolution using DCA models.
"""
import torch
import time
from typing import Dict, List, Tuple, Optional


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
    p_values: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Evolve sequences using unified GPU kernel (no split/merge overhead).
    
    MASSIVELY OPTIMIZED: All chains processed in parallel with mask-based logic.
    Eliminates split/merge overhead and enables full GPU parallelization.
    
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
        p_values: Pre-generated random values (n_chains,) for Metropolis/Gibbs split (optional)
        device: Torch device (CPU/GPU)
        dtype: Torch data type
        beta: Inverse temperature for Gibbs sampling
    
    Returns:
        Tuple of (evolved amino acid chains, evolved DNA chains, metro_time, gibbs_time)
    """
    N, L, q = chains.shape
    
    # Generate or use pre-generated random numbers for Metropolis/Gibbs split
    if p_values is None:
        random_values = torch.rand(N, device=device)
    else:
        random_values = p_values
    
    # Create boolean mask for Gibbs (True) vs Metropolis (False)
    use_gibbs = random_values > p  # Shape: (N,)
    
    # ========== UNIFIED KERNEL - ALL CHAINS IN PARALLEL ==========
    
    # Step 1: Randomly select one position per chain
    selected_sites = torch.randint(0, L, (N,), device=device) # (N,) e.g. [3, 0, 7, ..., L-1]
    batch_arange = torch.arange(N, device=device) # (N,) e.g. [0, 1, 2, ..., N-1]

    # Step 2: Extract biases and couplings for selected sites
    biases = params["bias"][selected_sites]  # (N, q)
    couplings_batch = params["coupling_matrix"][selected_sites]  # (N, q, L, q)
    
    # Step 3: Compute coupling term using optimized bmm (faster than einsum)
    # Reshape: (N, q, L, q) @ (N, L, q) -> (N, q, q) -> sum over last dim -> (N, q)
    # More efficient: (N, q, L*q) @ (N, L*q, 1) -> (N, q, 1) -> squeeze
    chains_flat = chains.reshape(N, L * q, 1)  # (N, L*q, 1)
    couplings_flat = couplings_batch.reshape(N, q, L * q)  # (N, q, L*q)
    coupling_term = torch.bmm(couplings_flat, chains_flat).squeeze(-1)  # (N, q)
    local_field = biases + coupling_term  # (N, q)
    
    # Step 4: Get current state
    current_codon_indices = dna_chains[batch_arange, selected_sites]  # (N,)
    current_aa_onehot = chains[batch_arange, selected_sites]  # (N, q)
    
    # ========== EXTRACT PARAMS ONCE (avoid repeated dict lookups) ==========
    gap_idx = params["gap_codon_idx"]
    non_gap_codon_tensor = params["non_gap_codon_tensor"]
    stop_codon_mask = params["stop_codon_mask"]
    codon_to_aa_onehot = params["codon_to_aa_onehot"]
    log_codon_usage = params["log_codon_usage"]
    codon_to_aa_idx = params["codon_to_aa_idx"]
    
    # ========== METROPOLIS LOGIC (gap insertion/deletion) ==========
    
    is_gap = (current_codon_indices == gap_idx)  # (N,)
    
    # METROPOLIS PROPOSAL RULES (symmetric):
    # - From gap (index 0): propose any of the 64 non-gap codons uniformly (p=1/64 each)
    # - From non-gap: propose gap (p=1/64) OR stay (p=63/64)
    # - Stop codons (indices 62, 63, 64) are proposed but REJECTED in acceptance
    
    # Number of non-gap codons (should be 64: 61 coding + 3 stop)
    num_non_gap = non_gap_codon_tensor.shape[0]  # Should be 64
    
    # For gap positions: propose random codon from ALL 64 non-gap codons (including stops)
    random_non_gap_idx = torch.randint(0, num_non_gap, (N,), device=device) # (N,) between 0 and 63
    random_non_gap_codon = non_gap_codon_tensor[random_non_gap_idx] # (N,) codon indices between 0 and 63
    
    # For non-gap positions: 1/64 chance gap, 63/64 chance stay
    rand_vals = torch.rand(N, device=device, dtype=dtype)
    propose_gap_from_nongap = rand_vals < (1.0 / 64.0)
    
    # Combine proposals efficiently (no clone, direct where chain)
    # Priority: is_gap → random, propose_gap → gap, else → current
    # Create gap_proposal dynamically (torch.full is extremely fast on GPU)
    gap_proposal = torch.full((N,), gap_idx, dtype=torch.long, device=device)
    metro_proposed_codon = torch.where(
        is_gap, 
        random_non_gap_codon,
        torch.where(propose_gap_from_nongap, gap_proposal, current_codon_indices)
    )
    
    # Reject stop codons BEFORE computing energy (use pre-computed mask - faster than isin)
    is_stop_codon = stop_codon_mask[metro_proposed_codon]  # Boolean indexing (molto più veloce)
    
    # If stop codon proposed, replace with current (will be rejected anyway)
    metro_proposed_codon = torch.where(is_stop_codon, current_codon_indices, metro_proposed_codon)
    
    # Convert to amino acids using pre-computed one-hot lookup (no one_hot() call)
    metro_proposed_aa_onehot = codon_to_aa_onehot[metro_proposed_codon]  # (N, q) direct lookup
    
    # Metropolis acceptance with codon usage bias
    # Acceptance = (codon_usage[new] / codon_usage[old]) * exp(-beta * delta_E)
    # - Symmetric proposal (1/64 all directions)
    # - Codon usage bias included in acceptance ratio
    # - Stop codons have usage=0.0, so they are automatically rejected
    
    delta_E = torch.sum((current_aa_onehot - metro_proposed_aa_onehot) * local_field, dim=-1)
    
    # Get codon usage for current and proposed codons
    current_codon_usage = codon_usage[current_codon_indices]  # (N,)
    proposed_codon_usage = codon_usage[metro_proposed_codon]  # (N,)
    
    # Codon usage ratio (stop codons already replaced with current, so no extra check needed)
    codon_usage_ratio = proposed_codon_usage / (current_codon_usage + 1e-10)
    
    # Metropolis acceptance: min(1, (usage_new/usage_old) * exp(-beta * delta_E))
    # Clamp fused with computation (no intermediate tensor)
    metro_acceptance_prob = torch.clamp(codon_usage_ratio * torch.exp(-beta * delta_E), 0.0, 1.0)
    metro_accept = torch.rand(N, device=device, dtype=dtype) < metro_acceptance_prob
    
    # ========== GIBBS LOGIC (codon-aware sampling) ==========
    
    # Select random nucleotide position (0, 1, or 2)
    nucleotide_positions = torch.randint(0, 3, (N,), device=device)
    
    # Get valid codon mask (N, num_codons) - single gather operation
    num_codons = codon_neighbor_codon_tensor.shape[0]  # e.g. 65
    valid_codon_mask = codon_neighbor_codon_tensor[current_codon_indices, nucleotide_positions]  # (N, num_codons)
    
    # Build codon->AA mapping with mask (avoid clone by using where)
    codon_aa_indices = torch.where(
        valid_codon_mask,
        codon_to_aa_idx.unsqueeze(0),  # Lazy broadcast, no clone
        -1
    )  # (N, num_codons)
    
    # Compute logits efficiently with fused operations
    # Direct indexing for valid codons (faster than gather + where)
    aa_indices_safe = torch.where(valid_codon_mask, codon_aa_indices, 0)
    codon_logits = (beta * local_field).gather(1, aa_indices_safe) + log_codon_usage  # Fused add
    
    # Set invalid codons to -inf (single where, no intermediate tensor)
    codon_logits = torch.where(valid_codon_mask, codon_logits, torch.tensor(float('-inf'), dtype=dtype, device=device))
    
    # Gumbel-Max sampling (faster than multinomial)
    gumbel_noise = -torch.log(-torch.log(torch.rand(N, num_codons, device=device, dtype=dtype) + 1e-10) + 1e-10)
    gibbs_proposed_codon = (codon_logits + gumbel_noise).argmax(dim=-1)
    
    # Convert to amino acids using pre-computed one-hot lookup (no one_hot() call)
    gibbs_proposed_aa_onehot = codon_to_aa_onehot[gibbs_proposed_codon]  # (N, q) direct lookup
    
    # ========== COMBINE RESULTS WITH MASKS ==========
    
    # Combine acceptance: Gibbs always accepts, Metropolis uses metro_accept
    accept_mutation = use_gibbs | metro_accept  # Single boolean mask (N,)
    
    # Cache unsqueeze operations for reuse (avoid recomputing)
    use_gibbs_2d = use_gibbs.unsqueeze(-1)  # (N, 1)
    accept_mutation_2d = accept_mutation.unsqueeze(-1)  # (N, 1)
    
    # Select proposed codon based on method (no intermediate tensor)
    proposed_codon = torch.where(use_gibbs, gibbs_proposed_codon, metro_proposed_codon)
    
    # Single where for codon (no nested where)
    final_codon = torch.where(accept_mutation, proposed_codon, current_codon_indices)
    
    # Single where for AA with lazy broadcasting (no expand)
    proposed_aa_onehot = torch.where(
        use_gibbs_2d,  # Reuse cached unsqueeze
        gibbs_proposed_aa_onehot,
        metro_proposed_aa_onehot
    )
    
    final_aa_onehot = torch.where(
        accept_mutation_2d,  # Reuse cached unsqueeze
        proposed_aa_onehot,
        current_aa_onehot
    )
    
    # Update chains in-place
    chains[batch_arange, selected_sites] = final_aa_onehot
    dna_chains[batch_arange, selected_sites] = final_codon


    # also output a tensor containing where you have selcted sites and the new aa 
    selected_sites_tensor = selected_sites.unsqueeze(-1)  # (N, 1)
    # not in one hot
    new_aa_tensor = final_aa_onehot.argmax(dim=-1)  # (N,)
    # put the tensor together in shape (N, 2)
    mutation_info_tensor = torch.cat((selected_sites_tensor, new_aa_tensor.unsqueeze(-1)), dim=-1)  # (N, 2)
    # You can return these tensors if needed for analysis
    
    # Return timing (0 since unified kernel)
    return chains, dna_chains, mutation_info_tensor

