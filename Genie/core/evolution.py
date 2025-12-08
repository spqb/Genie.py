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
    selected_sites = torch.randint(0, L, (N,), device=device)
    batch_arange = torch.arange(N, device=device)
    
    # Step 2: Extract biases and couplings for selected sites
    biases = params["bias"][selected_sites]  # (N, q)
    couplings_batch = params["coupling_matrix"][selected_sites]  # (N, q, L, q)
    
    # Step 3: Compute coupling term using einsum (fused)
    coupling_term = torch.einsum('nqLQ,nLQ->nq', couplings_batch, chains)
    local_field = biases + coupling_term  # (N, q)
    
    # Step 4: Get current state
    current_codon_indices = dna_chains[batch_arange, selected_sites]  # (N,)
    current_aa_onehot = chains[batch_arange, selected_sites]  # (N, q)
    
    # ========== METROPOLIS LOGIC (gap insertion/deletion) ==========
    gap_idx = params["gap_codon_idx"]  # Python int (pre-extracted in main.py)
    non_gap_codon_tensor = params["non_gap_codon_tensor"]
    codon_to_aa_idx = params["codon_to_aa_idx"]
    
    is_gap = (current_codon_indices == gap_idx)  # (N,)
    
    # CORRECTED PROPOSAL RULES:
    # - From gap: cannot propose gap itself (p=0), any of the 63 non-gap codons with p=1/63 each
    # - From non-gap: propose same codon with p=63/64, or gap with p=1/64
    
    # For gap positions: propose random NON-GAP codon from the 63 non-gap codons
    # non_gap_codon_tensor contains indices of all 63 non-gap codons
    num_non_gap = non_gap_codon_tensor.shape[0]  # Should be 63
    random_non_gap_idx = torch.randint(0, num_non_gap, (N,), device=device)
    random_non_gap_codon = non_gap_codon_tensor[random_non_gap_idx]
    
    # For non-gap positions: 1/64 chance gap, 63/64 chance stay
    rand_vals = torch.rand(N, device=device, dtype=dtype)
    propose_gap_from_nongap = rand_vals < (1.0 / 64.0)
    
    # Combine: if gap → random non-gap codon, if non-gap → gap (1/64) or stay (63/64)
    metro_proposed_codon = torch.where(
        is_gap,
        random_non_gap_codon,  # gap → any non-gap codon uniformly (p=1/63 each)
        torch.where(
            propose_gap_from_nongap,
            torch.full((N,), gap_idx, device=device, dtype=torch.long),  # non-gap → gap (p=1/64)
            current_codon_indices  # non-gap → stay (p=63/64)
        )
    )
    
    # Reject stop codons (>= 62)
    is_stop_codon = metro_proposed_codon >= 62
    metro_proposed_codon = torch.where(
        is_stop_codon,
        current_codon_indices,  # Replace stop with current (will be rejected anyway)
        metro_proposed_codon
    )
    
    # Convert to amino acids
    metro_proposed_aa_idx = codon_to_aa_idx[metro_proposed_codon]
    metro_proposed_aa_onehot = torch.nn.functional.one_hot(metro_proposed_aa_idx, num_classes=q).to(dtype)
    
    # Metropolis acceptance with proposal probability ratio
    # Since proposals are now ASYMMETRIC, we need to include q(reverse)/q(forward)
    # - gap → non-gap: q(forward) = 1/63, q(reverse) = 1/64 → ratio = (1/64)/(1/63) = 63/64
    # - non-gap → gap: q(forward) = 1/64, q(reverse) = 1/63 → ratio = (1/63)/(1/64) = 64/63
    # - non-gap → same: q(forward) = 63/64, q(reverse) = 63/64 → ratio = 1
    
    delta_E = torch.sum((current_aa_onehot - metro_proposed_aa_onehot) * local_field, dim=-1)
    
    # Calculate proposal probability ratio q(x'->x) / q(x->x')
    proposed_is_gap = (metro_proposed_codon == gap_idx)
    
    # Initialize ratio to 1 (for non-gap → same transitions)
    proposal_ratio = torch.ones(N, device=device, dtype=dtype)
    
    # gap → non-gap: multiply by 63/64
    proposal_ratio = torch.where(
        is_gap & ~proposed_is_gap,
        torch.full((N,), 63.0/64.0, device=device, dtype=dtype),
        proposal_ratio
    )
    
    # non-gap → gap: multiply by 64/63
    proposal_ratio = torch.where(
        ~is_gap & proposed_is_gap,
        torch.full((N,), 64.0/63.0, device=device, dtype=dtype),
        proposal_ratio
    )
    
    # Metropolis acceptance: min(1, exp(-beta * delta_E) * proposal_ratio)
    metro_acceptance_prob = torch.exp(-beta * delta_E) * proposal_ratio
    
    # Reject stop codons (>= 62)
    metro_acceptance_prob = torch.where(is_stop_codon, torch.zeros_like(metro_acceptance_prob), metro_acceptance_prob)
    
    # Clamp to [0, 1]
    metro_acceptance_prob = torch.clamp(metro_acceptance_prob, 0.0, 1.0)
    metro_accept = torch.rand(N, device=device, dtype=dtype) < metro_acceptance_prob
    
    # ========== GIBBS LOGIC (codon-aware sampling) ==========
    
    # Select random nucleotide position (0, 1, or 2)
    nucleotide_positions = torch.randint(0, 3, (N,), device=device)
    
    # Get valid codon mask (N, num_codons) - single gather operation
    num_codons = codon_neighbor_codon_tensor.shape[0]
    valid_codon_mask = codon_neighbor_codon_tensor[current_codon_indices, nucleotide_positions]
    
    # Pre-compute log codon usage (broadcast-friendly shape)
    log_codon_usage = torch.log(codon_usage + 1e-10)  # (num_codons,)
    
    # Build codon->AA mapping with mask (avoid clone by using where)
    codon_aa_indices = torch.where(
        valid_codon_mask,
        codon_to_aa_idx.unsqueeze(0),  # Lazy broadcast, no clone
        -1
    )
    
    # Compute logits directly with advanced indexing (fused operation)
    codon_logits = torch.full((N, num_codons), float('-inf'), dtype=dtype, device=device)
    valid_mask = valid_codon_mask  # Reuse existing mask, no need for >= 0 check
    
    # Direct gather without intermediate tensors
    aa_indices_safe = torch.where(valid_mask, codon_aa_indices, 0)
    gathered_aa_logits = (beta * local_field).gather(1, aa_indices_safe)
    
    # Use where instead of masked assignment for torch.compile compatibility
    logit_values = gathered_aa_logits + log_codon_usage
    codon_logits = torch.where(valid_mask, logit_values, codon_logits)
    
    # Gumbel-Max sampling (faster than multinomial)
    gumbel_noise = -torch.log(-torch.log(torch.rand(N, num_codons, device=device, dtype=dtype) + 1e-10) + 1e-10)
    gibbs_proposed_codon = (codon_logits + gumbel_noise).argmax(dim=-1)
    
    # Convert to amino acids (direct indexing, no intermediate tensor)
    gibbs_proposed_aa_idx = codon_aa_indices[batch_arange, gibbs_proposed_codon]
    gibbs_proposed_aa_onehot = torch.nn.functional.one_hot(gibbs_proposed_aa_idx, num_classes=q).to(dtype)
    
    # ========== COMBINE RESULTS WITH MASKS ==========
    
    # Combine acceptance: Gibbs always accepts, Metropolis uses metro_accept
    accept_mutation = use_gibbs | metro_accept  # Single boolean mask (N,)
    
    # Select proposed codon based on method (no intermediate tensor)
    proposed_codon = torch.where(use_gibbs, gibbs_proposed_codon, metro_proposed_codon)
    
    # Single where for codon (no nested where)
    final_codon = torch.where(accept_mutation, proposed_codon, current_codon_indices)
    
    # Single where for AA with lazy broadcasting (no expand)
    proposed_aa_onehot = torch.where(
        use_gibbs.unsqueeze(-1),  # Lazy broadcast to (N, q)
        gibbs_proposed_aa_onehot,
        metro_proposed_aa_onehot
    )
    
    final_aa_onehot = torch.where(
        accept_mutation.unsqueeze(-1),  # Lazy broadcast to (N, q)
        proposed_aa_onehot,
        current_aa_onehot
    )
    
    # Update chains in-place
    chains[batch_arange, selected_sites] = final_aa_onehot
    dna_chains[batch_arange, selected_sites] = final_codon
    
    # Return timing (0 since unified kernel)
    return chains, dna_chains, 0.0, 0.0

