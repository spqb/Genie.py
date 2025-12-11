"""
Reconstruct final chains from initial chains and mutation log.

This script takes the initial_chains.fasta and mutation_log.csv files
and reconstructs the final sequences by applying all mutations in order.
This is useful for validating the mutation logging system.

Usage:
    python scripts/reconstruct_chains.py <output_folder>
    
Example:
    python scripts/reconstruct_chains.py results/experiment_001
"""
import os
import sys
import argparse
import pandas as pd
from adabmDCA.fasta import import_from_fasta, write_fasta, get_tokens


def reconstruct_chains_from_log(initial_chains_file, mutation_log_file, tokens, output_file=None):
    """
    Reconstruct final chains by applying mutations from log to initial chains.
    
    Args:
        initial_chains_file: Path to initial_chains.fasta
        mutation_log_file: Path to mutation_log.csv
        tokens: List of amino acid tokens (alphabet)
        output_file: Path to save reconstructed chains (optional)
    
    Returns:
        reconstructed_sequences: List of reconstructed sequence strings
        headers: List of sequence headers
    """
    print(f"Loading initial chains from: {initial_chains_file}")
    headers, sequences_str = import_from_fasta(initial_chains_file, tokens, filter_sequences=True)
    
    # Convert sequences to list of lists (mutable) for easier mutation
    sequences = []
    for seq_str in sequences_str:
        # Convert string to list of characters
        if isinstance(seq_str, str):
            sequences.append(list(seq_str))
        else:
            # If already indices, convert to amino acid letters
            seq_chars = [tokens[idx] for idx in seq_str]
            sequences.append(seq_chars)
    
    N_chains = len(sequences)
    L = len(sequences[0])
    print(f"  Loaded {N_chains} chains of length {L}")
    
    # Create mapping from chain_id to sequence index
    # Headers are like "chain_0", "chain_1", etc.
    chain_id_to_index = {}
    for idx, header in enumerate(headers):
        # Extract chain_id from header (e.g., "chain_0" -> 0)
        if header.startswith("chain_"):
            chain_id = int(header.split("_")[1])
            chain_id_to_index[chain_id] = idx
        else:
            # If header doesn't follow pattern, assume it matches the index
            chain_id_to_index[idx] = idx
    
    print(f"  Chain ID mapping: {chain_id_to_index}")
    
    # Load mutation log
    print(f"Loading mutation log from: {mutation_log_file}")
    df_mutations = pd.read_csv(mutation_log_file)
    print(f"  Loaded {len(df_mutations)} mutation records")
    
    # Sort by iteration to apply mutations in order
    df_mutations = df_mutations.sort_values('iteration')
    
    # Apply mutations
    print(f"Applying mutations...")
    for idx, row in df_mutations.iterrows():
        chain_id = int(row['chain_id'])
        position = int(row['position'])
        new_aa = row['new_aa']
        
        # Map chain_id to actual sequence index
        if chain_id not in chain_id_to_index:
            print(f"Warning: Unknown chain_id {chain_id} at row {idx}")
            continue
        
        seq_index = chain_id_to_index[chain_id]
        
        # Validate indices
        if seq_index < 0 or seq_index >= N_chains:
            print(f"Warning: Invalid seq_index {seq_index} at row {idx}")
            continue
        if position < 0 or position >= L:
            print(f"Warning: Invalid position {position} at row {idx}")
            continue
        
        # Apply mutation
        sequences[seq_index][position] = new_aa
    
    print(f"  Applied all mutations successfully")
    
    # Convert back to strings
    reconstructed_sequences = ["".join(seq) for seq in sequences]
    
    # Save if output file specified
    if output_file is not None:
        print(f"Saving reconstructed chains to: {output_file}")
        write_fasta(output_file, headers, reconstructed_sequences)
        print(f"  Saved {N_chains} chains")
    
    return reconstructed_sequences, headers


def compare_chains(reconstructed_chains, reconstructed_headers, final_chains, final_headers, verbose=False):
    """
    Compare two sets of chains and report differences.
    
    Args:
        reconstructed_chains: List of reconstructed sequence strings
        reconstructed_headers: List of headers for reconstructed chains
        final_chains: List of actual final sequence strings  
        final_headers: List of headers for final chains
        verbose: If True, print detailed differences
    
    Returns:
        bool: True if all chains match perfectly
    """
    # Create dictionaries for easy lookup by header
    recon_dict = {header: seq for header, seq in zip(reconstructed_headers, reconstructed_chains)}
    final_dict = {header: seq for header, seq in zip(final_headers, final_chains)}
    
    # Check if same headers exist
    recon_set = set(reconstructed_headers)
    final_set = set(final_headers)
    
    if recon_set != final_set:
        print(f"ERROR: Different headers found")
        print(f"  Only in reconstructed: {recon_set - final_set}")
        print(f"  Only in final: {final_set - recon_set}")
        return False
    
    all_match = True
    total_mismatches = 0
    
    for header in sorted(recon_set):
        seq1 = recon_dict[header] if isinstance(recon_dict[header], str) else "".join(recon_dict[header])
        seq2 = final_dict[header] if isinstance(final_dict[header], str) else "".join(final_dict[header])
        
        if seq1 != seq2:
            all_match = False
            # Count mismatches
            mismatches = sum(1 for a, b in zip(seq1, seq2) if a != b)
            total_mismatches += mismatches
            
            if verbose:
                print(f"{header}: {mismatches} mismatches")
                # Show first few mismatches
                shown = 0
                for pos, (a, b) in enumerate(zip(seq1, seq2)):
                    if a != b:
                        print(f"  Position {pos}: {a} (reconstructed) vs {b} (actual)")
                        shown += 1
                        if shown >= 5:  # Limit output
                            if mismatches > 5:
                                print(f"  ... and {mismatches - 5} more mismatches")
                            break
    
    if all_match:
        print(f"✓ SUCCESS: All {len(recon_set)} chains match perfectly!")
        return True
    else:
        print(f"✗ FAILURE: {total_mismatches} total mismatches across chains")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct final chains from initial chains and mutation log"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to output folder containing initial_chains.fasta and mutation_log.csv"
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="protein",
        help="Alphabet type (default: protein)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed mismatch information"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save reconstructed chains to specified file (default: reconstructed_chains.fasta in output folder)"
    )
    
    args = parser.parse_args()
    
    # Get file paths
    initial_chains_file = os.path.join(args.output_folder, "initial_chains.fasta")
    mutation_log_file = os.path.join(args.output_folder, "mutation_log.csv")
    final_chains_file = os.path.join(args.output_folder, "final_chains.fasta")
    
    # Validate files exist
    if not os.path.exists(initial_chains_file):
        print(f"ERROR: Initial chains file not found: {initial_chains_file}")
        sys.exit(1)
    if not os.path.exists(mutation_log_file):
        print(f"ERROR: Mutation log file not found: {mutation_log_file}")
        sys.exit(1)
    
    # Get tokens
    tokens = get_tokens(args.alphabet)
    
    # Reconstruct chains
    output_file = args.save if args.save else os.path.join(args.output_folder, "reconstructed_chains.fasta")
    reconstructed_sequences, headers = reconstruct_chains_from_log(
        initial_chains_file,
        mutation_log_file,
        tokens,
        output_file=output_file
    )
    
    # Compare with final chains if available
    if os.path.exists(final_chains_file):
        print(f"\nComparing with actual final chains from: {final_chains_file}")
        final_headers, final_sequences_str = import_from_fasta(final_chains_file, tokens, filter_sequences=True)
        
        # Convert to strings if needed
        final_sequences = []
        for seq in final_sequences_str:
            if isinstance(seq, str):
                final_sequences.append(seq)
            else:
                final_sequences.append("".join([tokens[idx] for idx in seq]))
        
        success = compare_chains(reconstructed_sequences, headers, final_sequences, final_headers, verbose=args.verbose)
        
        if success:
            print("\n" + "="*80)
            print("VALIDATION SUCCESSFUL")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("VALIDATION FAILED")
            print("="*80)
            sys.exit(1)
    else:
        print(f"\nWarning: Final chains file not found: {final_chains_file}")
        print("Skipping comparison. Reconstruction completed.")


if __name__ == "__main__":
    main()
