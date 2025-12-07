"""
Genie 2.0 - Main Application
"""
import os
import sys
import time
import torch
import numpy as np
from adabmDCA import get_tokens, import_from_fasta, load_params
from .utils.parser import parse_arguments
from .utils.codon_utils import precompute_sampling_tensors
from .core.evolution import evolve_sequences


def translate_to_dna_uniform(
    chains: torch.Tensor,
    amino_to_codons: dict,
    codon_to_idx: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Translate amino acid sequences to DNA codon sequences.
    
    For each amino acid, selects the first codon from the list uniformly.
    
    Args:
        chains: Amino acid sequences as indices (N, L)
        amino_to_codons: Mapping from amino acid indices to codon lists
        codon_to_idx: Mapping from codon strings to codon indices
        device: Torch device
    
    Returns:
        DNA sequences as codon indices (N, L)
    """
    N, L = chains.shape
    
    # Create mapping from amino acid index to codon index
    aa_to_codon_idx = {}
    for aa_idx, codons in amino_to_codons.items():
        if len(codons) > 0:
            first_codon = codons[0]
            if first_codon in codon_to_idx:
                aa_to_codon_idx[aa_idx] = codon_to_idx[first_codon]
    
    # Vectorized translation
    chains_cpu = chains.cpu().numpy().flatten()
    dna_flat = np.array([aa_to_codon_idx.get(int(aa), 0) for aa in chains_cpu])
    dna_chains = torch.from_numpy(dna_flat.reshape(N, L)).long().to(device)
    
    return dna_chains


def main():
    """
    Main function for the Genie application.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if required input files exist
    if not os.path.isfile(args.path_sequences):
        print(f"Error: Sequences file not found: {args.path_sequences}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(args.path_params):
        print(f"Error: Parameters file not found: {args.path_params}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check GPU availability and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {device_name}")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU")
    
    # Set dtype
    dtype = torch.float32
    
    # Print configuration
    print("="*60)
    print(" "*20 + "GENIE 2.0")
    print("="*60)
    print()
    print("Configuration:")
    print("-" * 60)
    print(f"  Sequences file    : {args.path_sequences}")
    print(f"  Parameters file   : {args.path_params}")
    print(f"  Output folder     : {args.output}")
    print(f"  Device            : {device}")
    print(f"  Data type         : {dtype}")
    print("-" * 60)
    print()
    
    # Set alphabet
    tokens = get_tokens("protein")
    print(f"Alphabet set to protein: {len(tokens)} tokens")
    
    # Pre-compute all sampling tensors (codon mappings, neighbors, GPU tensors)
    print("Pre-computing sampling tensors...")
    sampling_data = precompute_sampling_tensors(tokens, device)
    print(f"  Codon network: {len(sampling_data['codon_neighbors'])} codons")
    print(f"  Neighbor tensor: {sampling_data['codon_neighbor_tensor'].shape}")
    print(f"  Mutation lookup: {sampling_data['mutation_lookup'].shape}")
    print()
    
    # Extract commonly used references
    codon_to_amino = sampling_data["codon_to_amino"]
    amino_to_codons = sampling_data["amino_to_codons"]
    
    # Load sequences from FASTA file
    print(f"Loading sequences from {args.path_sequences}...")
    _, sequences = import_from_fasta(args.path_sequences, tokens, filter_sequences=True)
    sequences  = sequences[:9000]
    print(f"Loaded {len(sequences)} sequences")
    
    # Ensure sequences are on the correct device
    if not isinstance(sequences, torch.Tensor):
        sequences = torch.tensor(sequences, dtype=torch.long)
    sequences = sequences.to(device)
    
    # Load parameters
    print(f"Loading parameters from {args.path_params}...")
    params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
    print(f"Parameters loaded successfully")
    
    # Add sampling tensors to params for easy access
    params.update(sampling_data)
    print("Sampling tensors added to parameters")
    print()
    
    # Convert sequences to tensors
    print("Converting sequences to tensors...")
    chains = sequences  # Shape: (N, L) - amino acid indices
    N, L = chains.shape
    q = len(tokens)
    
    # Translate amino acid sequences to DNA codons uniformly
    print("Translating amino acid sequences to DNA codons...")
    dna_chains = translate_to_dna_uniform(chains, amino_to_codons, sampling_data["codon_to_idx"], device)
    print(f"DNA sequences: {dna_chains.shape}")
    
    # Convert amino acid chains to one-hot encoding
    chains_onehot = torch.nn.functional.one_hot(chains.long(), num_classes=q).to(dtype).to(device)
    print(f"One-hot sequences: {chains_onehot.shape}")
    print()
    
    # Run evolution for 1000 iterations
    print("Running evolution for 1000 iterations...")
    print("-" * 60)
    
    num_iterations = 1_000
    current_chains = chains_onehot.clone()
    current_dna_chains = dna_chains.clone()
    
    # Warm-up run (to avoid GPU initialization overhead)
    _, _ = evolve_sequences(
        chains=current_chains,
        dna_chains=current_dna_chains,
        params=params,
        codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
        mutation_lookup=sampling_data["mutation_lookup"],
        num_options=sampling_data["num_options"],
        p=0.5,
        device=device,
        dtype=dtype,
        beta=1.0
    )
    
    # Timed evolution
    start_time = time.time()
    
    for iteration in range(num_iterations):
        current_chains, current_dna_chains = evolve_sequences(
            chains=current_chains,
            dna_chains=current_dna_chains,
            params=params,
            codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
            mutation_lookup=sampling_data["mutation_lookup"],
            num_options=sampling_data["num_options"],
            p=0.5,
            device=device,
            dtype=dtype,
            beta=1.0
        )
        
        # Print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (iteration + 1) / elapsed
            print(f"  Iteration {iteration + 1}/{num_iterations} - {rate:.2f} iter/sec")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("-" * 60)
    print(f"\nEvolution completed!")
    print(f"  Total iterations: {num_iterations}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per iteration: {total_time/num_iterations*1000:.2f} ms")
    print(f"  Iterations per second: {num_iterations/total_time:.2f}")
    print(f"  Final chains: {current_chains.shape}")
    print(f"  Final DNA chains: {current_dna_chains.shape}")
    print()
    
    print("Genie 2.0 completed successfully")
    

if __name__ == "__main__":
    main()
