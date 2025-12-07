"""
Benchmark script to test Pearson correlation convergence with Metropolis vs Mixed sampling.
"""
import os
import sys
import time
import argparse
import torch
import numpy as np
from adabmDCA import get_tokens, import_from_fasta, load_params

from Genie.utils.parser import parse_arguments
from Genie.utils.codon_utils import precompute_sampling_tensors
from Genie.core.evolution import evolve_sequences
from Genie.main import translate_to_dna_uniform
from Genie.utils.stats import compute_target_statistics, compute_correlations


def benchmark_convergence(p_value, num_iterations=500_000, num_chains=1000):
    """
    Benchmark convergence to target statistics.
    
    Args:
        p_value: Probability threshold (0=all Metropolis, 0.5=mixed)
        num_iterations: Number of iterations to run
        num_chains: Number of chains to use
    
    Returns:
        dict: Dictionary with convergence statistics
    """
    # Setup
    sequences_file = "example_data/example_chains.fasta"
    params_file = "example_data/example_params.dat"
    target_file = "example_data/CM_130530_MC.fasta"
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    dtype = torch.float32
    
    # Load data
    print("Loading data...")
    tokens = get_tokens("protein")
    sampling_data = precompute_sampling_tensors(tokens, device)
    
    # Load sequences
    _, sequences = import_from_fasta(sequences_file, tokens, filter_sequences=True)
    if not isinstance(sequences, torch.Tensor):
        sequences = torch.tensor(sequences, dtype=torch.long)
    sequences = sequences.to(device)
    
    # Load target sequences for statistics
    _, target_sequences = import_from_fasta(target_file, tokens, filter_sequences=True)
    if not isinstance(target_sequences, torch.Tensor):
        target_sequences = torch.tensor(target_sequences, dtype=torch.long)
    target_sequences = target_sequences.to(device)
    
    params = load_params(fname=params_file, tokens=tokens, device=device, dtype=dtype)
    params.update(sampling_data)
    
    # Prepare chains - replicate first sequence num_chains times
    q = len(tokens)
    L = sequences.shape[1]
    
    print(f"Replicating first sequence {num_chains} times...")
    first_sequence = sequences[0:1]  # Shape: (1, L)
    chains = first_sequence.repeat(num_chains, 1)  # Shape: (num_chains, L)
    
    # Compute target statistics
    print("Computing target statistics...")
    target_onehot = torch.nn.functional.one_hot(target_sequences.long(), num_classes=q).to(dtype).to(device)
    pi_target, pij_target = compute_target_statistics(target_onehot)
    
    # Prepare evolution
    amino_to_codons = sampling_data["amino_to_codons"]
    dna_chains = translate_to_dna_uniform(chains, amino_to_codons, sampling_data["codon_to_idx"], device)
    chains_onehot = torch.nn.functional.one_hot(chains.long(), num_classes=q).to(dtype).to(device)
    
    current_chains = chains_onehot.clone()
    current_dna_chains = dna_chains.clone()
    
    # Compute initial correlation
    initial_pearson = compute_correlations(pi_target, pij_target, current_chains)
    print(f"Initial Pearson correlation: {initial_pearson:.6f}")
    print()
    
    # Track progress
    check_interval = 10_000
    pearson_history = [initial_pearson]
    iteration_checkpoints = [0]
    
    # Run evolution
    print(f"Running {num_iterations} iterations with p={p_value}...")
    print("-" * 60)
    
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
        
        # Check correlation periodically
        if (iteration + 1) % check_interval == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            pearson = compute_correlations(pi_target, pij_target, current_chains)
            pearson_history.append(pearson)
            iteration_checkpoints.append(iteration + 1)
            
            elapsed = time.time() - start_time
            iter_per_sec = (iteration + 1) / elapsed
            
            print(f"  Iter {iteration + 1:6d}: Pearson = {pearson:.6f} ({iter_per_sec:.1f} iter/s)")
            
            # Check if target reached
            if pearson >= 0.9:
                print(f"\n✓ Target correlation 0.9 reached at iteration {iteration + 1}")
                break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    final_pearson = pearson_history[-1]
    final_iteration = iteration_checkpoints[-1]
    
    return {
        'p_value': p_value,
        'final_pearson': final_pearson,
        'final_iteration': final_iteration,
        'total_time': total_time,
        'pearson_history': pearson_history,
        'iteration_checkpoints': iteration_checkpoints,
        'reached_target': final_pearson >= 0.9
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark convergence to target Pearson correlation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=500_000,
        help='Maximum number of iterations to run'
    )
    parser.add_argument(
        '--num_chains',
        type=int,
        default=1000,
        help='Number of chains to use'
    )
    parser.add_argument(
        '--target',
        type=float,
        default=0.9,
        help='Target Pearson correlation to reach'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("CONVERGENCE BENCHMARK: Pearson Correlation >= 0.9")
    print("="*60)
    print()
    print("Configuration:")
    print(f"  Target: Pearson correlation >= {args.target}")
    print(f"  Max iterations: {args.num_iterations:,}")
    print(f"  Number of chains: {args.num_chains:,}")
    print(f"  Initial state: All chains = first sequence")
    print()
    print("="*60)
    print()
    
    # Test with p=0.0 (100% Metropolis)
    print("\n" + "="*60)
    print("TEST 1: p=0.0 (100% Metropolis)")
    print("="*60)
    print()
    result_metropolis = benchmark_convergence(p_value=0.0, num_iterations=args.num_iterations, num_chains=args.num_chains)
    
    # Test with p=0.5 (50% Metropolis, 50% Gibbs)
    print("\n" + "="*60)
    print("TEST 2: p=0.5 (50% Metropolis, 50% Gibbs)")
    print("="*60)
    print()
    result_mixed = benchmark_convergence(p_value=0.5, num_iterations=args.num_iterations, num_chains=args.num_chains)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print()
    
    print("Metropolis (p=0.0):")
    print(f"  Final Pearson:     {result_metropolis['final_pearson']:.6f}")
    print(f"  Final iteration:   {result_metropolis['final_iteration']:,}")
    print(f"  Total time:        {result_metropolis['total_time']:.2f}s")
    print(f"  Target reached:    {'✓ YES' if result_metropolis['reached_target'] else '✗ NO'}")
    print()
    
    print("Mixed (p=0.5):")
    print(f"  Final Pearson:     {result_mixed['final_pearson']:.6f}")
    print(f"  Final iteration:   {result_mixed['final_iteration']:,}")
    print(f"  Total time:        {result_mixed['total_time']:.2f}s")
    print(f"  Target reached:    {'✓ YES' if result_mixed['reached_target'] else '✗ NO'}")
    print()
    
    # Comparison
    if result_metropolis['reached_target'] and result_mixed['reached_target']:
        time_ratio = result_metropolis['total_time'] / result_mixed['total_time']
        iter_ratio = result_metropolis['final_iteration'] / result_mixed['final_iteration']
        
        print("COMPARISON:")
        if time_ratio < 1:
            print(f"  Metropolis reached target {1/time_ratio:.2f}x faster (time)")
        else:
            print(f"  Mixed reached target {time_ratio:.2f}x faster (time)")
        
        if iter_ratio < 1:
            print(f"  Metropolis reached target in {1/iter_ratio:.2f}x fewer iterations")
        else:
            print(f"  Mixed reached target in {iter_ratio:.2f}x fewer iterations")
    elif result_metropolis['reached_target']:
        print("WINNER: Metropolis (only method to reach target)")
    elif result_mixed['reached_target']:
        print("WINNER: Mixed (only method to reach target)")
    else:
        print("RESULT: Neither method reached target correlation of 0.9")
        if result_metropolis['final_pearson'] > result_mixed['final_pearson']:
            diff = result_metropolis['final_pearson'] - result_mixed['final_pearson']
            print(f"  Metropolis achieved higher correlation (+{diff:.6f})")
        else:
            diff = result_mixed['final_pearson'] - result_metropolis['final_pearson']
            print(f"  Mixed achieved higher correlation (+{diff:.6f})")
    
    print()


if __name__ == "__main__":
    main()
