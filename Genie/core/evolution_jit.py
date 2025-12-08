"""
JIT-compiled version of evolve_sequences for older PyTorch versions.
Alternative to torch.compile() for PyTorch < 2.0.

Use this if torch.compile() is not available.
"""
import torch
from typing import Dict, Tuple, Optional


@torch.jit.script
def evolve_sequences_jit(
    chains: torch.Tensor,
    dna_chains: torch.Tensor,
    bias: torch.Tensor,
    coupling_matrix: torch.Tensor,
    gap_codon_idx: torch.Tensor,
    non_gap_codon_tensor: torch.Tensor,
    codon_to_aa_idx: torch.Tensor,
    codon_neighbor_codon_tensor: torch.Tensor,
    codon_usage: torch.Tensor,
    p: float,
    p_values: torch.Tensor,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled unified evolution kernel.
    
    Note: Requires flattened params (no Dict support in TorchScript).
    
    Args:
        chains: (N, L, q) amino acid sequences
        dna_chains: (N, L) codon indices
        bias: (L, q) DCA biases
        coupling_matrix: (L, q, L, q) DCA couplings
        gap_codon_idx: Scalar tensor with gap codon index
        non_gap_codon_tensor: (num_non_gap,) non-gap codon indices
        codon_to_aa_idx: (num_codons,) codon to amino acid mapping
        codon_neighbor_codon_tensor: (num_codons, 3, num_codons) neighbor mask
        codon_usage: (num_codons,) usage frequencies
        p: Metropolis probability threshold
        p_values: (N,) pre-generated random values
        beta: Inverse temperature
    
    Returns:
        (evolved chains, evolved dna_chains)
    """
    N: int = chains.size(0)
    L: int = chains.size(1)
    q: int = chains.size(2)
    
    # Metropolis/Gibbs split
    use_gibbs = p_values > p
    
    # Select random position per chain
    selected_sites = torch.randint(0, L, (N,), device=chains.device)
    batch_arange = torch.arange(N, device=chains.device)
    
    # Extract local fields
    biases = bias[selected_sites]
    couplings_batch = coupling_matrix[selected_sites]
    coupling_term = torch.einsum('nqLQ,nLQ->nq', couplings_batch, chains)
    local_field = biases + coupling_term
    
    # Current state
    current_codon_indices = dna_chains[batch_arange, selected_sites]
    current_aa_onehot = chains[batch_arange, selected_sites]
    
    # Metropolis logic
    gap_idx = gap_codon_idx.item()
    is_gap = (current_codon_indices == gap_idx)
    num_non_gap = non_gap_codon_tensor.size(0)
    
    random_non_gap_idx = torch.randint(0, num_non_gap, (N,), device=chains.device)
    metro_proposed_codon = torch.where(
        is_gap,
        non_gap_codon_tensor[random_non_gap_idx],
        torch.full((N,), gap_idx, device=chains.device, dtype=torch.long)
    )
    
    metro_proposed_aa_idx = codon_to_aa_idx[metro_proposed_codon]
    metro_proposed_aa_onehot = torch.nn.functional.one_hot(metro_proposed_aa_idx, num_classes=q).to(chains.dtype)
    
    delta_E = torch.sum((current_aa_onehot - metro_proposed_aa_onehot) * local_field, dim=-1)
    metro_acceptance_prob = torch.exp(-beta * delta_E)
    metro_acceptance_prob = torch.where(~is_gap, metro_acceptance_prob / 64.0, metro_acceptance_prob)
    metro_acceptance_prob = torch.clamp(metro_acceptance_prob, 0.0, 1.0)
    metro_accept = torch.rand(N, device=chains.device, dtype=chains.dtype) < metro_acceptance_prob
    
    # Gibbs logic
    nucleotide_positions = torch.randint(0, 3, (N,), device=chains.device)
    num_codons = codon_neighbor_codon_tensor.size(0)
    valid_codon_mask = codon_neighbor_codon_tensor[current_codon_indices, nucleotide_positions]
    
    log_codon_usage = torch.log(codon_usage + 1e-10)
    codon_aa_indices = torch.where(
        valid_codon_mask,
        codon_to_aa_idx.unsqueeze(0),
        -1
    )
    
    codon_logits = torch.full((N, num_codons), float('-inf'), dtype=chains.dtype, device=chains.device)
    valid_mask = valid_codon_mask
    
    aa_indices_safe = torch.where(valid_mask, codon_aa_indices, 0)
    gathered_aa_logits = (beta * local_field).gather(1, aa_indices_safe)
    codon_logits[valid_mask] = (gathered_aa_logits + log_codon_usage)[valid_mask]
    
    gumbel_noise = -torch.log(-torch.log(torch.rand(N, num_codons, device=chains.device, dtype=chains.dtype) + 1e-10) + 1e-10)
    gibbs_proposed_codon = (codon_logits + gumbel_noise).argmax(dim=-1)
    
    gibbs_proposed_aa_idx = codon_aa_indices[batch_arange, gibbs_proposed_codon]
    gibbs_proposed_aa_onehot = torch.nn.functional.one_hot(gibbs_proposed_aa_idx, num_classes=q).to(chains.dtype)
    
    # Combine results
    accept_mutation = use_gibbs | metro_accept
    proposed_codon = torch.where(use_gibbs, gibbs_proposed_codon, metro_proposed_codon)
    final_codon = torch.where(accept_mutation, proposed_codon, current_codon_indices)
    
    proposed_aa_onehot = torch.where(
        use_gibbs.unsqueeze(-1),
        gibbs_proposed_aa_onehot,
        metro_proposed_aa_onehot
    )
    
    final_aa_onehot = torch.where(
        accept_mutation.unsqueeze(-1),
        proposed_aa_onehot,
        current_aa_onehot
    )
    
    # Update in-place
    chains[batch_arange, selected_sites] = final_aa_onehot
    dna_chains[batch_arange, selected_sites] = final_codon
    
    return chains, dna_chains
