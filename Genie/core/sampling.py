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
    gap_codon_idx: int,
    non_gap_codon_tensor: torch.Tensor,
    codon_to_aa_idx: torch.Tensor,
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
        gap_codon_idx: Index of gap codon '---' in all_codons list (pre-computed)
        non_gap_codon_tensor: Tensor of non-gap codon indices (pre-computed)
        codon_to_aa_idx: Tensor mapping codon index to amino acid index (pre-computed)
        device: Torch device (CPU/GPU)
        dtype: Torch data type
        beta: Inverse temperature (default: 1.0)
    
    Returns:
        Tuple of (updated amino acid chains, updated DNA chains)
    """
    N, L, q = chains.shape
    
    # Gap amino acid is always at index 0
    gap_aa_idx = 0
    
    # Use pre-computed tensors (no more reconstruction overhead!)
    num_non_gap_codons = non_gap_codon_tensor.shape[0]  # Should be 61
    
    # Randomly select one position for each chain
    batch_arange = torch.arange(N, device=device)
    selected_sites = torch.randint(0, L, (N,), device=device)
    
    # Get current codons and amino acids at selected sites
    current_codon_indices = dna_chains[batch_arange, selected_sites]  # Shape: (N,)
    current_aa_onehot = chains[batch_arange, selected_sites]  # Shape: (N, q)
    current_aa_indices = current_aa_onehot.argmax(dim=-1)  # Shape: (N,)
    
    # Determine which chains have gap at selected position
    is_gap_mask = (current_codon_indices == gap_codon_idx)  # Shape: (N,)
    
    # NEW PROPOSAL RULES:
    # - From gap: propose any codon (0-63, including gap and stop) with prob 1/64 each
    # - From non-gap: propose gap with prob 1/64, or stay with prob 63/64
    
    # Propose new codons
    proposed_codon_indices = torch.zeros(N, dtype=torch.long, device=device)
    proposal_probs = torch.ones(N, dtype=dtype, device=device)  # Track proposal probabilities
    
    # For gap positions: propose random codon uniformly from ALL 64 codons (including 3 stop codons)
    # Total codons: 61 standard + 1 gap + 3 stop = 65, but we use 64-codon model (gap=61, stops=61,62,63)
    # Actually: 61 non-gap + 1 gap = 62 total in our model, we extend to 64 for stop codons
    num_gaps = is_gap_mask.sum().item()
    if num_gaps > 0:
        # Propose uniformly from 0-63 (64 codons: 61 standard + gap + 2 implicit stops)
        random_codon_idx = torch.randint(0, 64, (num_gaps,), device=device)
        proposed_codon_indices[is_gap_mask] = random_codon_idx
        proposal_probs[is_gap_mask] = 1.0 / 64.0  # P(propose any codon | current=gap) = 1/64
    
    # For non-gap positions: propose gap with prob 1/64, or stay with prob 63/64
    num_non_gaps = (~is_gap_mask).sum().item()
    if num_non_gaps > 0:
        # Randomly decide: 1/64 chance of proposing gap, 63/64 chance of staying
        rand_vals = torch.rand(num_non_gaps, device=device, dtype=dtype)
        propose_gap_mask = rand_vals < (1.0 / 64.0)
        
        # Initialize with current codon (stay)
        proposed_codon_indices[~is_gap_mask] = current_codon_indices[~is_gap_mask]
        
        # For those proposing gap, set to gap_codon_idx
        non_gap_indices = torch.where(~is_gap_mask)[0]
        gap_proposal_indices = non_gap_indices[propose_gap_mask]
        proposed_codon_indices[gap_proposal_indices] = gap_codon_idx
        
        # Proposal probabilities
        proposal_probs[~is_gap_mask] = torch.where(
            propose_gap_mask,
            torch.tensor(1.0 / 64.0, dtype=dtype, device=device),  # P(gap | non-gap)
            torch.tensor(63.0 / 64.0, dtype=dtype, device=device)  # P(stay | non-gap)
        )
    
    # Convert proposed codons to amino acid indices using pre-computed tensor
    proposed_aa_indices = codon_to_aa_idx[proposed_codon_indices]  # Shape: (N,)
    
    # Create proposed amino acid one-hot
    proposed_aa_onehot = torch.nn.functional.one_hot(proposed_aa_indices, num_classes=q).to(dtype)
    
    # Compute energies using Metropolis criterion
    # Extract biases and couplings for selected sites
    biases = params["bias"][selected_sites]  # Shape: (N, q)
    couplings_batch = params["coupling_matrix"][selected_sites]  # Shape: (N, q, L, q)
    
    # Compute coupling term using einsum (more efficient than bmm)
    # einsum 'nqLQ,nLQ->nq': for each n, sum over L,Q of couplings[n,q,L,Q] * chains[n,L,Q]
    coupling_term = torch.einsum('nqLQ,nLQ->nq', couplings_batch, chains)  # (N, q)
    
    # Local field
    local_field = biases + coupling_term  # Shape: (N, q)
    
    # Energy difference: ΔE = (old - new) · local_field
    delta_E = torch.sum((current_aa_onehot - proposed_aa_onehot) * local_field, dim=-1)  # Shape: (N,)
    
    # Metropolis acceptance: P_accept = exp(-beta * ΔE)
    # NO division by 64 - proposal probabilities are already symmetric (both 1/64)
    acceptance_prob = torch.exp(-beta * delta_E)
    
    # Reject stop codons (indices 61, 62, 63 in 64-codon model)
    # In our current model: codons 0-60 are standard, 61 is gap
    # If proposed_codon >= 62, it's a stop codon → reject
    is_stop_codon = proposed_codon_indices >= 62
    acceptance_prob[is_stop_codon] = 0.0  # Always reject stop codons
    
    # Clip acceptance probability to [0, 1]
    acceptance_prob = torch.clamp(acceptance_prob, 0.0, 1.0)
    
    # Accept/reject
    random_uniform = torch.rand(N, device=device, dtype=dtype)
    accept_mask = (random_uniform < acceptance_prob)  # Shape: (N,)
    
    # Update chains in-place (no clone needed!)
    final_aa_onehot = torch.where(accept_mask.unsqueeze(-1), proposed_aa_onehot, current_aa_onehot)
    final_codon_indices = torch.where(accept_mask, proposed_codon_indices, current_codon_indices)
    
    chains[batch_arange, selected_sites] = final_aa_onehot
    dna_chains[batch_arange, selected_sites] = final_codon_indices
    
    return chains, dna_chains


def gibbs_step_batch(
    chains: torch.Tensor, 
    dna_chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    codon_neighbor_tensor: torch.Tensor,
    codon_neighbor_codon_tensor: torch.Tensor,
    mutation_lookup: torch.Tensor,
    num_options: torch.Tensor,
    codon_usage: torch.Tensor,
    codon_to_aa_idx: torch.Tensor,
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
        codon_neighbor_codon_tensor: Pre-computed codon neighbor accessibility (num_codons, 3, num_codons)
        mutation_lookup: Pre-computed codon mutations (num_codons, 3, q, max_neighbors)
        num_options: Count of valid options (num_codons, 3, q)
        codon_usage: Tensor (num_codons,) with codon usage frequencies
        codon_to_aa_idx: Tensor (num_codons,) mapping codon index to amino acid index
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
    
    # Compute coupling term using einsum (more efficient than bmm)
    coupling_term = torch.einsum('nqLQ,nLQ->nq', couplings_batch, chains)  # (N, q)
    
    # Compute logits for all amino acids
    logits_aa = beta * (biases + coupling_term)  # Shape: (N, q)
    
    # Get current codons
    current_codon_indices = dna_chains[batch_arange, selected_sites]  # Shape: (N,)
    
    # Get valid codon mask using codon_neighbor_codon_tensor
    # Shape: codon_neighbor_codon_tensor[current_codon, nuc_pos] -> (num_codons,) boolean mask
    num_codons = codon_neighbor_codon_tensor.shape[0]
    valid_codon_mask = codon_neighbor_codon_tensor[current_codon_indices, nucleotide_positions]  # (N, num_codons)
    
    # Build codon-to-amino-acid mapping tensor (needed to map codon logits to aa logits)
    # We need to know which amino acid each codon codes for
    # This can be inferred from codon_neighbor_tensor: for each codon, find which aa it produces
    # codon_neighbor_tensor[codon_idx, :, :] has shape (3, q)
    # We can use the pattern: a codon maps to the aa that appears in all 3 positions
    # Actually, we need to build this from mutation_lookup or pass it explicitly
    # For now, we'll use a simpler approach: for each valid codon, find its amino acid
    # by checking which amino acid appears in codon_neighbor_tensor[codon_idx, 0, :]
    
    # Get amino acid for each codon by checking position 0 accessibility
    # A codon codes for the amino acid that is accessible at all positions (0,1,2)
    # Simpler: use the fact that codon_neighbor_tensor[codon_idx, pos, aa_idx] = True 
    # means mutating position 'pos' can reach 'aa_idx'
    # But to get the CURRENT amino acid of a codon, we need different info
    
    # Alternative: build codon_to_aa mapping from mutation_lookup
    # mutation_lookup[codon_idx, nuc_pos, aa_idx, :] contains codons that result from mutation
    # But we need the inverse: given a codon, what aa does it code for?
    
    # Most direct: use the current chains to get the amino acid
    current_aa_onehot = chains[batch_arange, selected_sites]  # Shape: (N, q)
    current_aa_indices = current_aa_onehot.argmax(dim=-1)  # Shape: (N,)
    
    # Now, for each valid codon, compute its logit
    # logit_codon[i] = logit_aa[corresponding_aa] * codon_usage[i]
    # We need to map each codon to its amino acid
    # Build this mapping: for each codon index, which aa index does it correspond to?
    
    # Use codon_neighbor_tensor to infer: a codon at index i produces amino acid j
    # if starting from that codon and mutating position 0,1,2 can reach different amino acids
    # But the codon itself encodes one specific amino acid
    
    # Better approach: use mutation_lookup in reverse
    # For each codon c and each position p, mutation_lookup[c, p, :, :] shows reachable codons
    # The amino acid the codon c itself codes for is not directly stored
    
    # Practical solution: build codon_to_aa_idx tensor from codon_neighbor_tensor
    # For a codon at index c: it codes for amino acid a if all three nucleotide positions
    # when NOT mutated, correspond to amino acid a
    # Actually, we can use the fact that chains[batch_arange, selected_sites] gives current aa
    
    # Let's build codon_to_aa mapping once (should be precomputed, but we'll do it here)
    # For each codon, the amino acid it codes for can be found by:
    # - For position 0: codon_neighbor_tensor[codon_idx, 0, :] shows which aa are accessible
    # - The codon's own aa is the one that appears when we DON'T mutate
    # We can infer this from the structure: if we have all_codons list and codon_to_amino dict
    # But those aren't passed to this function
    
    # Workaround: compute codon_to_aa_idx from valid neighbors
    # For each codon c: codon_neighbor_tensor[c, pos, aa] tells if mutating pos can reach aa
    # The identity (no mutation) should also be encoded somehow
    # Actually, codon_neighbor_tensor includes the CURRENT amino acid in accessible set
    
    # Simplified approach for now: 
    # For each batch element, we know current_codon -> current_aa
    # For valid codons from mutation, we can check mutation_lookup to find their aa
    
    # Let's build it differently: 
    # valid_codon_mask[n, c] = True means codon c is accessible from current_codon[n]
    # For each accessible codon c, find which aa it codes for
    # We can use codon_neighbor_tensor: the aa that codon c codes for is the one
    # where codon_neighbor_tensor[c, pos, aa] might be True for mutations
    
    # Most efficient: precompute codon_to_aa_idx as a tensor of shape (num_codons,)
    # For now, use an approximation: build it from codon_neighbor_tensor
    # codon_neighbor_tensor[codon_idx, 0, :] gives accessible aa from position 0
    # The codon's own aa should be inferable
    
    # Quick solution: since we're in a batch, for each chain we know current aa
    # For valid codons, use mutation_lookup to trace back
    # mutation_lookup[current_codon, nuc_pos, aa_idx, :] contains resulting codons
    # So if codon c is in mutation_lookup[current_codon, nuc_pos, aa_idx, :], then c codes for aa_idx
    
    # Build codon -> aa mapping for valid codons using pre-computed tensor
    # codon_to_aa_idx[codon_idx] gives the amino acid index for that codon
    # We need to build (N, num_codons) tensor mapping valid codons to their aa
    # For each batch element, only codons accessible via mutation are valid
    
    # Start with global codon->aa mapping, then mask by valid_codon_mask
    # Shape: (num_codons,) -> broadcast to (N, num_codons)
    codon_aa_indices = codon_to_aa_idx.unsqueeze(0).expand(N, -1).clone()  # (N, num_codons)
    
    # Mark invalid codons with -1
    codon_aa_indices[~valid_codon_mask] = -1
    
    # Compute logits for each valid codon
    # logit_codon[n, c] = logit_aa[n, aa_of_c] * log(codon_usage[c])
    # Use log(codon_usage) to keep logits in log-space
    log_codon_usage = torch.log(codon_usage + 1e-10)  # Add small epsilon to avoid log(0)
    
    # Broadcast: logits_aa[n, aa] -> logits_codon[n, c] based on codon_aa_indices[n, c]
    # For each (n, c), get logits_aa[n, codon_aa_indices[n, c]]
    codon_logits = torch.full((N, num_codons), float('-inf'), dtype=dtype, device=device)
    valid_mask = codon_aa_indices >= 0
    
    # Gather aa logits for valid codons
    # Use advanced indexing: for each valid (n, c), get logits_aa[n, codon_aa_indices[n, c]]
    batch_indices = batch_arange.unsqueeze(1).expand(-1, num_codons)  # (N, num_codons)
    aa_indices_for_codons = codon_aa_indices.clone()
    aa_indices_for_codons[~valid_mask] = 0  # Placeholder for invalid codons
    
    gathered_aa_logits = logits_aa[batch_indices, aa_indices_for_codons]  # (N, num_codons)
    
    # Compute codon logits: aa_logit + log(codon_usage)
    codon_logits[valid_mask] = gathered_aa_logits[valid_mask] + log_codon_usage[None, :].expand(N, -1)[valid_mask]
    
    # Sample new codon using Gumbel-Max trick (faster than multinomial on GPU)
    # Gumbel-Max: sample ~ Categorical(softmax(logits)) is equivalent to argmax(logits + Gumbel noise)
    gumbel_noise = -torch.log(-torch.log(torch.rand(N, num_codons, device=device, dtype=dtype) + 1e-10) + 1e-10)
    new_codon_indices = (codon_logits + gumbel_noise).argmax(dim=-1)  # Shape: (N,)
    
    # Get corresponding amino acid indices
    new_aa_indices = codon_aa_indices[batch_arange, new_codon_indices]  # Shape: (N,)
    
    # Create new one-hot residues
    new_residues = torch.nn.functional.one_hot(new_aa_indices, num_classes=q).to(dtype)
    
    # Update chains in-place (no clone needed!)
    chains[batch_arange, selected_sites] = new_residues
    dna_chains[batch_arange, selected_sites] = new_codon_indices
    
    return chains, dna_chains
