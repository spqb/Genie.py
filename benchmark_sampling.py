"""
Benchmark script to compare Metropolis vs Gibbs sampling speed.
"""
import os
import sys
import time
import torch
from adabmDCA import get_tokens, import_from_fasta, load_params

from Genie.utils.parser import parse_arguments
from Genie.utils.codon_utils import precompute_sampling_tensors
from Genie.core.evolution import evolve_sequences
from Genie.main import translate_to_dna_uniform


def benchmark_sampling(p_value, num_iterations=1000):
    """
    Benchmark evolution with specific p value.
    
    Args:
        p_value: Probability threshold (0=all Metropolis, 1=all Gibbs)
        num_iterations: Number of iterations to run
    
    Returns:
        float: Time elapsed in seconds
    """
    # Setup
    sequences_file = "example_data/example_chains.fasta"
    params_file = "example_data/example_params.dat"
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    dtype = torch.float32
    
    # Load data
    tokens = get_tokens("protein")
    sampling_data = precompute_sampling_tensors(tokens, device)
    
    _, sequences = import_from_fasta(sequences_file, tokens, filter_sequences=True)
    if not isinstance(sequences, torch.Tensor):
        sequences = torch.tensor(sequences, dtype=torch.long)
    sequences = sequences.to(device)
    
    params = load_params(fname=params_file, tokens=tokens, device=device, dtype=dtype)
    params.update(sampling_data)
    
    # Prepare chains
    chains = sequences
    N, L = chains.shape
    q = len(tokens)
    
    amino_to_codons = sampling_data["amino_to_codons"]
    dna_chains = translate_to_dna_uniform(chains, amino_to_codons, sampling_data["codon_to_idx"], device)
    chains_onehot = torch.nn.functional.one_hot(chains.long(), num_classes=q).to(dtype).to(device)
    
    current_chains = chains_onehot.clone()
    current_dna_chains = dna_chains.clone()
    
    # Warm-up (to ensure GPU is ready)
    for _ in range(10):
        current_chains, current_dna_chains = evolve_sequences(
            chains=current_chains,
            dna_chains=current_dna_chains,
            params=params,
            codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
            mutation_lookup=sampling_data["mutation_lookup"],
            num_options=sampling_data["num_options"],
            p=p_value,
            device=device,
            dtype=dtype,
            beta=1.0
        )
    
    # Synchronize GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for iteration in range(num_iterations):
        current_chains, current_dna_chains = evolve_sequences(
            chains=current_chains,
            dna_chains=current_dna_chains,
            params=params,
            codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
            mutation_lookup=sampling_data["mutation_lookup"],
            num_options=sampling_data["num_options"],
            p=p_value,
            device=device,
            dtype=dtype,
            beta=1.0
        )
    
    # Synchronize GPU after timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    return elapsed


def main():
    print("="*60)
    print("BENCHMARK: Metropolis vs Gibbs Sampling Speed")
    print("="*60)
    print()
    
    num_iterations = 100
    
    print(f"Running {num_iterations} iterations with p=0.0 (100% Metropolis)...")
    time_metropolis = benchmark_sampling(p_value=0.0, num_iterations=num_iterations)
    print(f"Time elapsed: {time_metropolis:.4f} seconds")
    print(f"Iterations per second: {num_iterations / time_metropolis:.2f}")
    print()
    
    print(f"Running {num_iterations} iterations with p=0.5 (50% Metropolis, 50% Gibbs)...")
    time_mixed = benchmark_sampling(p_value=0.5, num_iterations=num_iterations)
    print(f"Time elapsed: {time_mixed:.4f} seconds")
    print(f"Iterations per second: {num_iterations / time_mixed:.2f}")
    print()
    
    print(f"Running {num_iterations} iterations with p=1.0 (100% Gibbs)...")
    time_gibbs = benchmark_sampling(p_value=1.0, num_iterations=num_iterations)
    print(f"Time elapsed: {time_gibbs:.4f} seconds")
    print(f"Iterations per second: {num_iterations / time_gibbs:.2f}")
    print()
    
    print("="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Metropolis (p=0.0): {time_metropolis:.4f}s ({num_iterations / time_metropolis:.2f} iter/s)")
    print(f"Mixed (p=0.5):      {time_mixed:.4f}s ({num_iterations / time_mixed:.2f} iter/s)")
    print(f"Gibbs (p=1.0):      {time_gibbs:.4f}s ({num_iterations / time_gibbs:.2f} iter/s)")
    print()
    
    speedup_mg = time_metropolis / time_gibbs
    speedup_mm = time_metropolis / time_mixed
    speedup_gm = time_gibbs / time_mixed
    
    print("SPEEDUP COMPARISON:")
    if speedup_mg > 1:
        print(f"  Gibbs vs Metropolis:  Gibbs is {speedup_mg:.2f}x faster")
    else:
        print(f"  Gibbs vs Metropolis:  Metropolis is {1/speedup_mg:.2f}x faster")
    
    if speedup_mm > 1:
        print(f"  Mixed vs Metropolis:  Mixed is {speedup_mm:.2f}x faster")
    else:
        print(f"  Mixed vs Metropolis:  Metropolis is {1/speedup_mm:.2f}x faster")
    
    if speedup_gm > 1:
        print(f"  Gibbs vs Mixed:       Gibbs is {speedup_gm:.2f}x faster")
    else:
        print(f"  Gibbs vs Mixed:       Mixed is {1/speedup_gm:.2f}x faster")
    print()


if __name__ == "__main__":
    main()
