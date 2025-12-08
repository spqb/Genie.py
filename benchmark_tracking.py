"""
Benchmark script to track Pearson correlation evolution starting from example_chains.
No target - just tracks how correlation evolves over iterations.
"""
import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from adabmDCA import get_tokens, import_from_fasta, load_params

from Genie.utils.parser import parse_arguments
from Genie.utils.codon_utils import precompute_sampling_tensors
from Genie.core.evolution import evolve_sequences
from Genie.main import translate_to_dna_uniform
from Genie.utils.stats import compute_target_statistics, compute_correlations


def track_pearson_evolution(p_value, num_iterations, num_chains=None):
    """
    Track Pearson correlation evolution starting from example_chains.
    
    Args:
        p_value: Probability threshold (0=all Metropolis, 0.5=mixed, 1=all Gibbs)
        num_iterations: Number of iterations to run
        num_chains: Number of chains to use (None = use all)
    
    Returns:
        dict: Dictionary with tracking results
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
    
    # Prepare chains
    q = len(tokens)
    L = sequences.shape[1]
    
    # Use specified number of chains or all sequences
    if num_chains is not None and num_chains < sequences.shape[0]:
        print(f"Using first {num_chains} sequences from example_chains.fasta...")
        chains = sequences[:num_chains]
    else:
        chains = sequences
        num_chains = chains.shape[0]
        print(f"Using all {num_chains} sequences from example_chains.fasta...")
    
    # Compute target statistics
    print("Computing target statistics from CM_130530_MC.fasta...")
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
    check_interval = max(1, num_iterations // 100)  # Check ~100 times
    pearson_history = [initial_pearson]
    iteration_checkpoints = [0]
    
    # Run evolution
    print(f"Running {num_iterations:,} iterations with p={p_value}...")
    print("-" * 60)
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        current_chains, current_dna_chains = evolve_sequences(
            chains=current_chains,
            dna_chains=current_dna_chains,
            params=params,
            codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
            codon_neighbor_codon_tensor=sampling_data["codon_neighbor_codon_tensor"],
            mutation_lookup=sampling_data["mutation_lookup"],
            num_options=sampling_data["num_options"],
            codon_usage=sampling_data["codon_usage"],
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
            
            if (iteration + 1) % (check_interval * 10) == 0 or iteration == num_iterations - 1:
                print(f"  Iter {iteration + 1:7,}: Pearson = {pearson:.6f} ({iter_per_sec:.1f} iter/s)")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    final_pearson = pearson_history[-1]
    pearson_change = final_pearson - initial_pearson
    
    return {
        'p_value': p_value,
        'initial_pearson': initial_pearson,
        'final_pearson': final_pearson,
        'pearson_change': pearson_change,
        'num_iterations': num_iterations,
        'num_chains': num_chains,
        'total_time': total_time,
        'pearson_history': pearson_history,
        'iteration_checkpoints': iteration_checkpoints
    }


def main():
    parser = argparse.ArgumentParser(
        description="Track Pearson correlation evolution from example_chains",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=10_000,
        help='Number of iterations to run'
    )
    parser.add_argument(
        '--num_chains',
        type=int,
        default=None,
        help='Number of chains to use (default: all sequences)'
    )
    parser.add_argument(
        '-p', '--p_values',
        type=float,
        nargs='+',
        default=[0.0, 0.5, 1.0],
        help='List of p values to test (e.g., 0.0 0.5 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pearson_tracking',
        help='Output directory for plots and results'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("PEARSON CORRELATION TRACKING")
    print("="*60)
    print()
    print("Configuration:")
    print(f"  Iterations: {args.num_iterations:,}")
    print(f"  Chains: {'All sequences' if args.num_chains is None else args.num_chains}")
    print(f"  P values: {args.p_values}")
    print(f"  Starting from: example_chains.fasta")
    print(f"  Target stats: CM_130530_MC.fasta")
    print()
    print("="*60)
    print()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run for each p value
    results = []
    for p_value in args.p_values:
        print("\n" + "="*60)
        print(f"TEST: p={p_value}")
        print("="*60)
        print()
        
        result = track_pearson_evolution(
            p_value=p_value,
            num_iterations=args.num_iterations,
            num_chains=args.num_chains
        )
        results.append(result)
        
        print()
        print(f"Results for p={p_value}:")
        print(f"  Initial Pearson: {result['initial_pearson']:.6f}")
        print(f"  Final Pearson:   {result['final_pearson']:.6f}")
        print(f"  Change:          {result['pearson_change']:+.6f}")
        print(f"  Time:            {result['total_time']:.2f}s")
        print()
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    plt.figure(figsize=(12, 6))
    
    for result in results:
        p_val = result['p_value']
        label = f"p={p_val}"
        if p_val == 0.0:
            label += " (100% Metropolis)"
        elif p_val == 1.0:
            label += " (100% Gibbs)"
        elif p_val == 0.5:
            label += " (50/50 Mixed)"
        
        plt.plot(
            result['iteration_checkpoints'],
            result['pearson_history'],
            linewidth=2,
            label=label,
            marker='o',
            markersize=3,
            alpha=0.8
        )
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Pearson Correlation', fontsize=12)
    plt.title('Pearson Correlation Evolution with Target Statistics', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(args.output, 'pearson_evolution_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")
    
    # Save numerical results
    results_file = os.path.join(args.output, 'pearson_tracking_results.txt')
    with open(results_file, 'w') as f:
        f.write("PEARSON CORRELATION TRACKING RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Iterations: {args.num_iterations:,}\n")
        f.write(f"  Chains: {results[0]['num_chains']}\n\n")
        
        for result in results:
            f.write(f"\np={result['p_value']}:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Initial Pearson: {result['initial_pearson']:.6f}\n")
            f.write(f"  Final Pearson:   {result['final_pearson']:.6f}\n")
            f.write(f"  Change:          {result['pearson_change']:+.6f}\n")
            f.write(f"  Time:            {result['total_time']:.2f}s\n")
            f.write(f"  Iter/sec:        {result['num_iterations']/result['total_time']:.1f}\n")
        
        f.write("\n\nDETAILED HISTORY:\n")
        f.write("="*60 + "\n")
        for result in results:
            f.write(f"\np={result['p_value']}:\n")
            f.write("Iteration\tPearson\n")
            for iter_num, pearson in zip(result['iteration_checkpoints'], result['pearson_history']):
                f.write(f"{iter_num}\t{pearson:.6f}\n")
    
    print(f"Results saved to {results_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        p_val = result['p_value']
        change = result['pearson_change']
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  p={p_val}: {result['initial_pearson']:.6f} {direction} {result['final_pearson']:.6f} ({change:+.6f})")
    print()


if __name__ == "__main__":
    main()
