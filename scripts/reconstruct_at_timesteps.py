"""
Reconstruct chains at specific timesteps from initial chains and mutation log.

This function takes initial chains and mutation log, and returns a tensor
containing the sequences at specified timesteps.

Usage:
    from scripts.reconstruct_at_timesteps import (
        reconstruct_at_timesteps,
        save_reconstructed_timesteps_fasta,
    )
    from adabmDCA.fasta import get_tokens, import_from_fasta
    
    sequences = reconstruct_at_timesteps(
        initial_chains_file="path/to/initial_chains.fasta",
        mutation_log_file="path/to/mutation_log.csv",
        timesteps=[0, 100, 500, 1000],
        alphabet="protein"
    )
    
    # sequences.shape: (len(timesteps), n_chains, L)
    tokens = get_tokens("protein")
    headers, _ = import_from_fasta("path/to/initial_chains.fasta", tokens, filter_sequences=True)
    save_reconstructed_timesteps_fasta(
        sequences_tensor=sequences,
        timesteps=[0, 100, 500, 1000],
        headers=headers,
        output_dir="path/to/reconstructed_timesteps",
        alphabet="protein"
    )
"""
import torch
import os
import pandas as pd
from adabmDCA.fasta import import_from_fasta, get_tokens, write_fasta


def reconstruct_at_timesteps(
    initial_chains_file: str,
    mutation_log_file: str,
    timesteps: list,
    alphabet: str = "protein"
) -> torch.Tensor:
    """
    Reconstruct sequences at specific timesteps.
    
    Args:
        initial_chains_file: Path to initial_chains.fasta
        mutation_log_file: Path to mutation_log.csv
        timesteps: List of iteration numbers at which to extract sequences
                  (e.g., [0, 100, 200, 500])
        alphabet: Alphabet type (default: "protein")
    
    Returns:
        torch.Tensor of shape (len(timesteps), n_chains, L) containing
        sequences encoded as integer indices according to alphabet
        
    Example:
        >>> sequences = reconstruct_at_timesteps(
        ...     "results/initial_chains.fasta",
        ...     "results/mutation_log.csv",
        ...     [0, 100, 500, 1000]
        ... )
        >>> sequences.shape
        torch.Size([4, 10, 96])  # 4 timesteps, 10 chains, length 96
    """
    # Get tokens
    tokens = get_tokens(alphabet)
    
    # Load initial chains
    headers, sequences_data = import_from_fasta(initial_chains_file, tokens, filter_sequences=True)
    
    # Convert to list of lists (mutable) for easier mutation
    sequences = []
    for seq_str in sequences_data:
        if isinstance(seq_str, str):
            sequences.append(list(seq_str))
        else:
            # If already indices, convert to amino acid letters
            seq_chars = [tokens[idx] for idx in seq_str]
            sequences.append(seq_chars)
    
    N_chains = len(sequences)
    L = len(sequences[0])
    
    # Create mapping from chain_id to sequence index
    # Headers are like "chain_0", "chain_1", etc.
    chain_id_to_index = {}
    for idx, header in enumerate(headers):
        if header.startswith("chain_"):
            chain_id = int(header.split("_")[1])
            chain_id_to_index[chain_id] = idx
        else:
            chain_id_to_index[idx] = idx
    
    # Load mutation log
    df_mutations = pd.read_csv(mutation_log_file)
    
    # Sort by iteration to apply mutations in order
    df_mutations = df_mutations.sort_values('iteration')
    
    # Validate that requested timesteps are available in mutation log
    # Note: mutation log saves iterations 0-indexed, so iteration N is saved as N-1
    available_timesteps_in_log = sorted(df_mutations['iteration'].unique())
    available_timesteps = [t + 1 for t in available_timesteps_in_log]  # Convert to 1-indexed
    requested_timesteps = sorted(set(timesteps))
    
    # Remove timestep 0 from validation (it's the initial state, always available)
    timesteps_to_check = [t for t in requested_timesteps if t > 0]
    
    # Check if all requested timesteps (except 0) are in mutation log
    missing_timesteps = [t for t in timesteps_to_check if t not in available_timesteps]
    
    if missing_timesteps:
        available_str = ", ".join(map(str, available_timesteps[:10]))
        if len(available_timesteps) > 10:
            available_str += f", ... ({len(available_timesteps)} total)"
        raise ValueError(
            f"Requested timesteps {missing_timesteps} are not available in mutation log.\n"
            f"Available timesteps in mutation log: {available_str}\n"
            f"Note: Timestep 0 (initial state) is always available without being in the log.\n"
            f"Note: Mutation log saves iteration N as N-1 (0-indexed)."
        )
    
    # Ensure timesteps are sorted and include 0 if initial state is requested
    timesteps_sorted = sorted(set(timesteps))
    
    # Create token to index mapping for fast conversion
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    
    # Initialize output tensor
    output_tensor = torch.zeros(len(timesteps_sorted), N_chains, L, dtype=torch.long)
    
    # Track which timestep we're at
    timestep_idx = 0
    current_timestep = timesteps_sorted[timestep_idx]
    
    # If timestep 0 is requested, save initial state
    if current_timestep == 0:
        for chain_idx in range(N_chains):
            for pos in range(L):
                aa_letter = sequences[chain_idx][pos]
                output_tensor[timestep_idx, chain_idx, pos] = token_to_idx[aa_letter]
        timestep_idx += 1
        if timestep_idx < len(timesteps_sorted):
            current_timestep = timesteps_sorted[timestep_idx]
    
    # Apply mutations and save at requested timesteps
    mutation_idx = 0
    total_mutations = len(df_mutations)
    
    while mutation_idx < total_mutations and timestep_idx < len(timesteps_sorted):
        row = df_mutations.iloc[mutation_idx]
        mutation_iteration = int(row['iteration']) + 1  # Convert from 0-indexed to 1-indexed
        
        # Check if we've reached the next timestep
        if mutation_iteration > current_timestep:
            # Save current state at this timestep
            for chain_idx in range(N_chains):
                for pos in range(L):
                    aa_letter = sequences[chain_idx][pos]
                    output_tensor[timestep_idx, chain_idx, pos] = token_to_idx[aa_letter]
            
            # Move to next timestep
            timestep_idx += 1
            if timestep_idx < len(timesteps_sorted):
                current_timestep = timesteps_sorted[timestep_idx]
            else:
                break
        else:
            # Apply this mutation
            chain_id = int(row['chain_id'])
            position = int(row['position'])
            new_aa = row['new_aa']
            
            # Map chain_id to actual sequence index
            if chain_id in chain_id_to_index:
                seq_index = chain_id_to_index[chain_id]
                if 0 <= seq_index < N_chains and 0 <= position < L:
                    sequences[seq_index][position] = new_aa
            
            mutation_idx += 1
    
    # Save remaining timesteps (after all mutations)
    while timestep_idx < len(timesteps_sorted):
        for chain_idx in range(N_chains):
            for pos in range(L):
                aa_letter = sequences[chain_idx][pos]
                output_tensor[timestep_idx, chain_idx, pos] = token_to_idx[aa_letter]
        timestep_idx += 1
    
    return output_tensor


def save_reconstructed_timesteps_fasta(
    sequences_tensor: torch.Tensor,
    timesteps: list,
    headers: list,
    output_dir: str,
    alphabet: str = "protein"
) -> list:
    """
    Save reconstructed timestep tensors as one FASTA file per timestep.

    Args:
        sequences_tensor: Tensor of shape (len(timesteps), n_chains, L)
        timesteps: Timesteps corresponding to the first tensor dimension
        headers: FASTA headers in the same chain order as sequences_tensor
        output_dir: Directory where FASTA files will be written
        alphabet: Alphabet type (default: "protein")

    Returns:
        List of written FASTA file paths.
    """
    tokens = get_tokens(alphabet)
    os.makedirs(output_dir, exist_ok=True)

    if sequences_tensor.shape[0] != len(timesteps):
        raise ValueError(
            f"Expected one tensor slice per timestep, got "
            f"{sequences_tensor.shape[0]} slices and {len(timesteps)} timesteps."
        )

    if sequences_tensor.shape[1] != len(headers):
        raise ValueError(
            f"Expected one header per chain, got "
            f"{len(headers)} headers and {sequences_tensor.shape[1]} chains."
        )

    headers = [str(header) for header in headers]
    written_files = []

    for timestep_idx, timestep in enumerate(timesteps):
        sequences = [
            "".join(tokens[idx.item()] for idx in sequences_tensor[timestep_idx, chain_idx])
            for chain_idx in range(sequences_tensor.shape[1])
        ]
        output_file = os.path.join(output_dir, f"reconstructed_t{timestep}.fasta")
        write_fasta(output_file, headers, sequences)
        written_files.append(output_file)

    return written_files


def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reconstruct sequences at specific timesteps"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to output folder containing initial_chains.fasta and mutation_log.csv"
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        required=True,
        help="Comma-separated list of timesteps (e.g., '0,100,500,1000')"
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="protein",
        help="Alphabet type (default: protein)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save tensor (optional, .pt format)"
    )
    parser.add_argument(
        "--save-fasta",
        action="store_true",
        help="Save one FASTA file per requested timestep"
    )
    parser.add_argument(
        "--fasta-output-dir",
        type=str,
        default=None,
        help="Directory for reconstructed FASTA files (default: <output_folder>/reconstructed_timesteps)"
    )
    
    args = parser.parse_args()
    
    # Parse timesteps
    timesteps = sorted(set(int(x.strip()) for x in args.timesteps.split(',')))
    
    # Get file paths
    initial_chains_file = os.path.join(args.output_folder, "initial_chains.fasta")
    mutation_log_file = os.path.join(args.output_folder, "mutation_log.csv")
    
    # Validate files exist
    if not os.path.exists(initial_chains_file):
        print(f"ERROR: Initial chains file not found: {initial_chains_file}")
        return 1
    if not os.path.exists(mutation_log_file):
        print(f"ERROR: Mutation log file not found: {mutation_log_file}")
        return 1
    
    print(f"Reconstructing sequences at timesteps: {timesteps}")
    print(f"Initial chains: {initial_chains_file}")
    print(f"Mutation log: {mutation_log_file}")
    print()
    
    # Reconstruct
    sequences_tensor = reconstruct_at_timesteps(
        initial_chains_file,
        mutation_log_file,
        timesteps,
        args.alphabet
    )
    
    print(f"✓ Reconstruction complete")
    print(f"  Tensor shape: {sequences_tensor.shape}")
    print(f"  (timesteps={sequences_tensor.shape[0]}, chains={sequences_tensor.shape[1]}, length={sequences_tensor.shape[2]})")
    
    # Save if requested
    if args.output is not None:
        torch.save(sequences_tensor, args.output)
        print(f"  Saved to: {args.output}")

    if args.save_fasta:
        headers, _ = import_from_fasta(initial_chains_file, get_tokens(args.alphabet), filter_sequences=True)
        fasta_output_dir = args.fasta_output_dir
        if fasta_output_dir is None:
            fasta_output_dir = os.path.join(args.output_folder, "reconstructed_timesteps")
        written_files = save_reconstructed_timesteps_fasta(
            sequences_tensor=sequences_tensor,
            timesteps=timesteps,
            headers=headers,
            output_dir=fasta_output_dir,
            alphabet=args.alphabet
        )
        print(f"  Saved FASTA files to: {fasta_output_dir}")
        for path in written_files:
            print(f"    {path}")
    
    # Show a sample
    print(f"\nSample (first chain, first 10 positions at each timestep):")
    tokens = get_tokens(args.alphabet)
    for t_idx, timestep in enumerate(timesteps):
        sample_seq = "".join([tokens[idx.item()] for idx in sequences_tensor[t_idx, 0, :10]])
        print(f"  t={timestep:>6}: {sample_seq}...")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
