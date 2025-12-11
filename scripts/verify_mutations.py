#!/usr/bin/env python3
"""
Mutation Verification Script for Genie and Genie-AA

This script verifies the correctness of the mutation logging by:
1. Loading initial chains from FASTA file
2. Applying all mutations from the mutation log step-by-step
3. Comparing the reconstructed chains with the saved final chains

Usage:
    python verify_mutations.py <output_folder> [--alphabet protein]

Example:
    python verify_mutations.py test_genie_pf14_10kchains
    python verify_mutations.py test_genie_aa_output --alphabet protein
"""

import argparse
import sys
import os
from typing import Dict, List, Tuple


def parse_fasta(filepath: str) -> Dict[str, str]:
    """
    Parse a FASTA file and return a dictionary mapping sequence IDs to sequences.
    
    Args:
        filepath: Path to the FASTA file
        
    Returns:
        Dictionary {sequence_id: sequence_string}
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                # Start new sequence
                current_id = line[1:].split()[0]  # Take first word after >
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def load_mutation_log(filepath: str) -> List[Tuple[int, int, int, str]]:
    """
    Load mutation log from CSV file.
    
    Args:
        filepath: Path to mutation_log.csv
        
    Returns:
        List of tuples (iteration, chain_id, position, new_aa)
    """
    mutations = []
    
    with open(filepath, 'r') as f:
        header = f.readline()  # Skip header
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            iteration = int(parts[0])
            chain_id = int(parts[1])
            position = int(parts[2])
            new_aa = parts[3]
            
            mutations.append((iteration, chain_id, position, new_aa))
    
    return mutations


def apply_mutations(
    initial_chains: Dict[str, str], 
    mutations: List[Tuple[int, int, int, str]]
) -> Dict[str, str]:
    """
    Apply mutations to initial chains.
    
    Args:
        initial_chains: Dictionary {chain_id: sequence}
        mutations: List of (iteration, chain_id, position, new_aa)
        
    Returns:
        Dictionary of reconstructed chains after applying all mutations
    """
    # Convert sequences to mutable lists
    chains = {}
    for chain_id, seq in initial_chains.items():
        chains[chain_id] = list(seq)
    
    # Apply mutations in order
    for iteration, chain_idx, position, new_aa in mutations:
        chain_id = f"chain_{chain_idx}"
        if chain_id in chains:
            chains[chain_id][position] = new_aa
    
    # Convert back to strings
    return {chain_id: ''.join(seq) for chain_id, seq in chains.items()}


def verify_chains(
    reconstructed: Dict[str, str], 
    final_chains: Dict[str, str]
) -> Tuple[bool, List[str]]:
    """
    Verify that reconstructed chains match final chains.
    
    Args:
        reconstructed: Chains reconstructed by applying mutations
        final_chains: Chains loaded from final_chains.fasta
        
    Returns:
        Tuple of (all_match, list_of_mismatches)
    """
    mismatches = []
    
    # Check all chains in final_chains
    for chain_id, final_seq in final_chains.items():
        if chain_id not in reconstructed:
            mismatches.append(f"Chain {chain_id}: missing in reconstructed chains")
            continue
        
        recon_seq = reconstructed[chain_id]
        
        if final_seq != recon_seq:
            # Find first difference
            for i, (a, b) in enumerate(zip(final_seq, recon_seq)):
                if a != b:
                    mismatches.append(
                        f"Chain {chain_id}: mismatch at position {i} "
                        f"(expected '{a}', got '{b}')"
                    )
                    break
            else:
                # Length difference
                mismatches.append(
                    f"Chain {chain_id}: length mismatch "
                    f"(expected {len(final_seq)}, got {len(recon_seq)})"
                )
    
    # Check for extra chains in reconstructed
    for chain_id in reconstructed:
        if chain_id not in final_chains:
            mismatches.append(f"Chain {chain_id}: extra chain in reconstructed")
    
    return len(mismatches) == 0, mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Verify mutation log correctness for Genie/Genie-AA"
    )
    parser.add_argument(
        "output_folder",
        help="Path to the output folder containing initial_chains.fasta, "
             "mutation_log.csv, and final_chains.fasta"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # File paths
    initial_file = os.path.join(args.output_folder, "initial_chains.fasta")
    mutation_file = os.path.join(args.output_folder, "mutation_log.csv")
    final_file = os.path.join(args.output_folder, "final_chains.fasta")
    
    # Check files exist
    for filepath, desc in [
        (initial_file, "Initial chains"),
        (mutation_file, "Mutation log"),
        (final_file, "Final chains")
    ]:
        if not os.path.exists(filepath):
            print(f"ERROR: {desc} file not found: {filepath}")
            sys.exit(1)
    
    print("=" * 70)
    print("  MUTATION VERIFICATION TEST")
    print("=" * 70)
    print(f"\n  Output folder: {args.output_folder}\n")
    
    # Load initial chains
    print("[1/4] Loading initial chains...")
    initial_chains = parse_fasta(initial_file)
    print(f"      Loaded {len(initial_chains)} chains")
    if args.verbose and initial_chains:
        first_id = list(initial_chains.keys())[0]
        print(f"      First chain length: {len(initial_chains[first_id])}")
    
    # Load mutation log
    print("[2/4] Loading mutation log...")
    mutations = load_mutation_log(mutation_file)
    print(f"      Loaded {len(mutations):,} mutations")
    if args.verbose and mutations:
        num_iterations = mutations[-1][0] + 1
        print(f"      Iterations: {num_iterations:,}")
        print(f"      Mutations per iteration: ~{len(mutations) // num_iterations}")
    
    # Apply mutations
    print("[3/4] Applying mutations step-by-step...")
    reconstructed = apply_mutations(initial_chains, mutations)
    print(f"      Reconstructed {len(reconstructed)} chains")
    
    # Load final chains
    print("[4/4] Loading and verifying final chains...")
    final_chains = parse_fasta(final_file)
    print(f"      Loaded {len(final_chains)} final chains")
    
    # Verify
    all_match, mismatches = verify_chains(reconstructed, final_chains)
    
    print("\n" + "-" * 70)
    
    if all_match:
        print("\n  ✓ VERIFICATION PASSED!")
        print(f"    All {len(final_chains)} chains match correctly.")
        print("\n    The mutation log correctly reconstructs the final chains")
        print("    starting from the initial chains.")
        print("\n" + "=" * 70 + "\n")
        sys.exit(0)
    else:
        print("\n  ✗ VERIFICATION FAILED!")
        print(f"\n    Found {len(mismatches)} mismatches:\n")
        for i, msg in enumerate(mismatches[:10]):  # Show first 10
            print(f"    {i+1}. {msg}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
