"""
Genie 2.0 - Main Application
"""
import os
import sys
import time
import torch
import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from adabmDCA import get_tokens, import_from_fasta, load_params
from adabmDCA.statmech import compute_energy
from adabmDCA.dataset import DatasetDCA

from .utils.parser import parse_arguments
from .utils.codon_utils import precompute_sampling_tensors
from .core.evolution import evolve_sequences
from .utils.pca_utils import train_pca, project_sequences, plot_pca_evolution, plot_pca_with_density
from .utils.stats import compute_target_statistics, compute_correlations

# Compile evolve_sequences for maximum performance
# Mode options: 'default', 'reduce-overhead', 'max-autotune'
# max-autotune: extensive autotuning, 2-3x faster (compiles once, 10-30s warmup)
# fullgraph=False: allows graph breaks for Python scalar operations
try:
    evolve_sequences_compiled = torch.compile(
        evolve_sequences,
        mode='max-autotune',
        fullgraph=False,  # Allow graph breaks for scalar operations
        dynamic=False
    )
    COMPILE_AVAILABLE = True
except Exception as e:
    print(f"Warning: torch.compile() not available ({e}). Using uncompiled version.")
    evolve_sequences_compiled = evolve_sequences
    COMPILE_AVAILABLE = False


def translate_to_dna_uniform(
    chains: torch.Tensor,
    amino_to_codons_tensor: torch.Tensor,
    amino_to_num_codons: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Translate amino acid sequences to DNA codon sequences (GPU-optimized).
    
    For each amino acid, randomly selects a codon uniformly from available options.
    Fully vectorized on GPU - no CPU loops!
    
    Args:
        chains: Amino acid sequences as indices (N, L)
        amino_to_codons_tensor: Tensor (num_aa, max_codons) mapping aa → codon indices
        amino_to_num_codons: Tensor (num_aa,) with count of codons per amino acid
        device: Torch device
    
    Returns:
        DNA sequences as codon indices (N, L)
    """
    N, L = chains.shape
    
    # Flatten chains for vectorized operations
    chains_flat = chains.flatten()  # Shape: (N*L,) on GPU
    
    # For each amino acid, get the number of available codons
    num_codons_per_position = amino_to_num_codons[chains_flat]  # Shape: (N*L,)
    
    # Generate random indices for codon selection (uniform within available codons)
    random_codon_indices = torch.rand(N * L, device=device) * num_codons_per_position.float()
    random_codon_indices = random_codon_indices.long()  # Shape: (N*L,)
    
    # Gather the selected codon indices using advanced indexing
    # amino_to_codons_tensor[chains_flat[i], random_codon_indices[i]]
    selected_codons = amino_to_codons_tensor[chains_flat, random_codon_indices]  # Shape: (N*L,)
    
    # Reshape back to (N, L)
    dna_chains = selected_codons.reshape(N, L)
    
    return dna_chains


def main():
    """
    Main function for the Genie application.
    """
    start_time_total = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if required input files exist
    if args.path_chains is not None and not os.path.isfile(args.path_chains):
        print(f"Error: Sequences file not found: {args.path_chains}", file=sys.stderr)
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
    if args.path_chains is not None:
        print(f"  Sequences file    : {args.path_chains}")
    else:
        print(f"  Sequences         : Random initialization")
        print(f"  Num chains        : {args.num_chains}")
        print(f"  Sequence length   : auto-detect from DCA params")
    print(f"  Parameters file   : {args.path_params}")
    print(f"  Output folder     : {args.output}")
    print(f"  Device            : {device}")
    print(f"  Data type         : {dtype}")
    print(f"  P (Metropolis)    : {args.p_metropolis} (Metropolis/Gibbs ratio)")
    print("-" * 60)
    print()
    
    # Set alphabet
    t0 = time.time()
    tokens = get_tokens("protein")
    print(f"Alphabet set to protein: {len(tokens)} tokens (took {time.time()-t0:.2f}s)")
    
    # Pre-compute all sampling tensors (codon mappings, neighbors, GPU tensors)
    print("Pre-computing sampling tensors...")
    t0 = time.time()
    sampling_data = precompute_sampling_tensors(tokens, device)
    elapsed = time.time() - t0
    print(f"  Codon network: {len(sampling_data['codon_neighbors'])} codons")
    print(f"  Neighbor tensor: {sampling_data['codon_neighbor_tensor'].shape}")
    print(f"  Mutation lookup: {sampling_data['mutation_lookup'].shape}")
    print(f"  (took {elapsed:.2f}s)")
    print()
    
    # Extract commonly used references
    codon_to_amino = sampling_data["codon_to_amino"]
    amino_to_codons = sampling_data["amino_to_codons"]
    
    # Load parameters first to get sequence length
    print(f"Loading parameters from {args.path_params}...")
    t0 = time.time()
    params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
    print(f"Parameters loaded successfully (took {time.time()-t0:.2f}s)")
    
    # Get sequence length from DCA parameters
    L_params = params["bias"].shape[0]  # bias shape is (L, q)
    print(f"  DCA model sequence length: {L_params}")
    
    # Load or initialize sequences
    if args.path_chains is not None:
        # Load sequences from FASTA file
        print(f"Loading sequences from {args.path_chains}...")
        t0 = time.time()
        _, sequences = import_from_fasta(args.path_chains, tokens, filter_sequences=True)
        print(f"Loaded {len(sequences)} sequences (took {time.time()-t0:.2f}s)")
        
        # Ensure sequences are on the correct device
        if not isinstance(sequences, torch.Tensor):
            sequences = torch.tensor(sequences, dtype=torch.long)
        sequences = sequences.to(device)
        
        chains = sequences # Shape: (N, L) - amino acid indices
        N, L = chains.shape
        
        # Verify sequence length matches DCA model
        if L != L_params:
            print(f"Warning: Loaded sequences have length {L}, but DCA model expects {L_params}")
    else:
        # Initialize random sequences using DCA model length
        L = L_params
        N = args.num_chains
        print(f"Initializing {N} random sequences of length {L}...")
        # Random initialization excluding gap (token 0)
        chains = torch.randint(1, len(tokens), (N, L), dtype=torch.long, device=device)
        print(f"Random sequences initialized: {chains.shape}")
    
    q = len(tokens)
    
    # Add sampling tensors to params for easy access
    params.update(sampling_data)
    print("Sampling tensors added to parameters")
    print()
    
    # If num_chains is provided and sequences were loaded from file, select subset
    if args.num_chains is not None and args.path_chains is not None:
        if args.num_chains <= 0:
            print(f"Error: num_chains must be positive, got {args.num_chains}", file=sys.stderr)
            sys.exit(1)
        if args.num_chains > N:
            print(f"Warning: num_chains {args.num_chains} is larger than available sequences {N}. Using all {N} sequences.")
        else:
            print(f"Selecting first {args.num_chains} sequences out of {N}...")
            # extract randomly num_chains sequences
            indices = torch.randperm(N)[:args.num_chains]
            chains = chains[indices]
            N = args.num_chains
            print(f"Using {N} sequences for evolution")
            print()
    
    # If seq_index is provided, replicate that single sequence N times
    if args.seq_index is not None:
        if args.path_chains is None:
            print(f"Error: --seq_index can only be used with --path_chains", file=sys.stderr)
            sys.exit(1)
        if args.seq_index < 0 or args.seq_index >= chains.shape[0]:
            print(f"Error: seq_index {args.seq_index} is out of range [0, {chains.shape[0]-1}]", file=sys.stderr)
            sys.exit(1)
        
        print(f"Replicating sequence {args.seq_index} {N} times...")
        selected_sequence = chains[args.seq_index:args.seq_index+1]  # Shape: (1, L)
        chains = selected_sequence.repeat(N, 1)  # Shape: (N, L)
        print(f"All {N} sequences are now identical to sequence {args.seq_index}")
        print()
    
    # Train PCA on reference dataset
    pca_data_path = args.pca_data
    clustering_seqid = 0.8 
    no_reweighting = False
    print(f"Training PCA on reference dataset ({pca_data_path})...")
    t0 = time.time()
    if os.path.isfile(pca_data_path):
        dataset =  dataset = DatasetDCA(
            path_data=pca_data_path,
            path_weights=None,
            alphabet=tokens,
            clustering_th=clustering_seqid,
            no_reweighting=no_reweighting,
            filter_sequences=True,
            remove_duplicates=True,
            device=device,
            dtype=dtype,
            message=False,
        )
        
        pca_sequences = dataset.data  # (N_ref, L) tensor of amino acid indices 
        pca_weights = dataset.weights.squeeze()  # (N_ref,) tensor of sequence weights
        print("pca_weights:", pca_weights.shape)
        if not isinstance(pca_sequences, torch.Tensor):
            pca_sequences = torch.tensor(pca_sequences, dtype=torch.long)
        pca_sequences = pca_sequences.to(device)
        
        # Convert to one-hot encoding
        pca_sequences_onehot = pca_sequences #torch.nn.functional.one_hot(pca_sequences.long(), num_classes=q).to(dtype).to(device)
        print(f"PCA training data: {pca_sequences_onehot.shape}")
        
        # Train PCA (with automatic subsampling if N > 5000)
        pca, pca_subsample_indices = train_pca(pca_sequences_onehot, n_components=2, weights=pca_weights, max_samples=5000)
        print(f"PCA trained: {pca.n_components} components, explained variance: {pca.explained_variance_ratio_}")
        
        # Compute target statistics for correlation tracking
        print("Computing target statistics...")
       
        pseudocount = 1. / dataset.weights.sum().item()
        pi_target, pij_target = compute_target_statistics(pca_sequences_onehot, weights=pca_weights, pseudo_count=pseudocount)
        print(f"PCA and statistics computed (took {time.time()-t0:.2f}s)")
        print()
    else:
        print(f"Warning: PCA training data not found at {pca_data_path}. Skipping PCA analysis.")
        pca = None
        pca_subsample_indices = None
        pi_target = None
        pij_target = None
        print()
    
    # Translate amino acid sequences to DNA codons uniformly
    print("Translating amino acid sequences to DNA codons...")
    t0 = time.time()
    dna_chains = translate_to_dna_uniform(
        chains, 
        sampling_data["amino_to_codons_tensor"], 
        sampling_data["amino_to_num_codons"], 
        device
    )
    print(f"DNA sequences: {dna_chains.shape} (took {time.time()-t0:.2f}s)")
    
    # Convert amino acid chains to one-hot encoding
    chains_onehot = torch.nn.functional.one_hot(chains.long(), num_classes=q).to(dtype).to(device)
    print(f"One-hot sequences: {chains_onehot.shape}")
    print()
    
    # Run evolution
    num_iterations = args.num_iterations
    print(f"Running evolution for {num_iterations} iterations...")
    print("-" * 60)
    t_evolution_start = time.time()
    current_chains = chains_onehot.clone()
    current_dna_chains = dna_chains.clone()
    
    # Save initial state
    initial_chains_aa = current_chains.argmax(dim=-1)  # Convert one-hot to indices (N, L)
    initial_dna_chains = current_dna_chains.clone()
    
    # Track statistics every 100 iterations
    correlation_history = []
    iteration_checkpoints = []
    
    # Compute initial correlation if target statistics are available
    if pi_target is not None and pij_target is not None:
        initial_pearson = compute_correlations(pi_target, pij_target, current_chains)
        correlation_history.append(initial_pearson)
        iteration_checkpoints.append(0)
        print(f"Initial Pearson correlation: {initial_pearson:.6f}")
        print()
    
    # Run evolution
    total_metropolis_time = 0.0
    total_gibbs_time = 0.0
    
    # Pre-generate random numbers for better GPU efficiency (batch of 1000)
    random_batch_size = min(1000, num_iterations)
    random_batch = torch.rand(random_batch_size, N, device=device)
    random_batch_idx = 0
    
    # Use compiled version if available
    evolve_fn = evolve_sequences_compiled if COMPILE_AVAILABLE else evolve_sequences
    if COMPILE_AVAILABLE:
        print("Using torch.compile() with mode='max-autotune'")
        print("First iteration will be slower due to compilation (10-30s)...\n")
    
    for iteration in range(num_iterations):
        if (iteration + 1) % 10000 == 0:
            elapsed = time.time() - t_evolution_start
            iter_per_sec = (iteration + 1) / elapsed
            print(f"  Iteration {iteration + 1}/{num_iterations}... ({iter_per_sec:.1f} iter/s)")
        
        # Detailed timing for first iteration to identify bottlenecks
        if iteration == 0:
            t_evolve = time.time()
        
        # Get pre-generated random value for this iteration
        p_values = random_batch[random_batch_idx]
        random_batch_idx += 1
        
        # Regenerate random batch when exhausted
        if random_batch_idx >= random_batch_size and iteration + 1 < num_iterations:
            remaining = num_iterations - (iteration + 1)
            random_batch_size = min(1000, remaining)
            random_batch = torch.rand(random_batch_size, N, device=device)
            random_batch_idx = 0
        
        current_chains, current_dna_chains, metro_time, gibbs_time = evolve_fn(
            chains=current_chains,
            dna_chains=current_dna_chains,
            params=params,
            codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
            codon_neighbor_codon_tensor=sampling_data["codon_neighbor_codon_tensor"],
            mutation_lookup=sampling_data["mutation_lookup"],
            num_options=sampling_data["num_options"],
            codon_usage=sampling_data["codon_usage"],
            p=args.p_metropolis,
            p_values=p_values,  # Pre-generated random values
            device=device,
            dtype=dtype,
            beta=1.0
        )
        
        total_metropolis_time += metro_time
        total_gibbs_time += gibbs_time
        
        if iteration == 0:
            first_evolve_time = time.time() - t_evolve
            print(f"  First evolve_sequences call: {first_evolve_time:.4f}s")
            print(f"    - Metropolis: {metro_time:.4f}s")
            print(f"    - Gibbs: {gibbs_time:.4f}s")
        
        # Print correlation every 100 iterations
        if (iteration + 1) % 10000 == 0 and pi_target is not None and pij_target is not None:
            t_corr = time.time()
            pearson = compute_correlations(pi_target, pij_target, current_chains)
            correlation_history.append(pearson)
            iteration_checkpoints.append(iteration + 1)
            
            # Compute gap frequencies (pi_target[:, 0] is gap frequency at each position)
            gap_freq_target = pi_target[:, 0].cpu()  # Shape: (L,)
            pi_chains, _ = compute_target_statistics(current_chains)
            gap_freq_current = pi_chains[:, 0].cpu() # Shape: (L,)
            gap_freq_diff = np.abs(gap_freq_target - gap_freq_current)
            max_gap_diff = gap_freq_diff.max()
            mean_gap_diff = gap_freq_diff.mean()
            pearson_gap = np.corrcoef(gap_freq_target, gap_freq_current)[0,1]
            
            corr_time = time.time() - t_corr
            print(f"    Iteration {iteration + 1}: Pearson = {pearson:.6f} | Gap freq diff: max={max_gap_diff:.4f}, mean={mean_gap_diff:.4f}, Pearson gap = {pearson_gap:.6f} (time: {corr_time:.4f}s)")
    
    # Get final state
    final_chains_aa = current_chains.argmax(dim=-1)  # Convert one-hot to indices (N, L)
    final_dna_chains = current_dna_chains.clone()
    
    # Compute energies before and after evolution
    print("\nComputing energies...")
    t0 = time.time()
    initial_chains_onehot = torch.nn.functional.one_hot(initial_chains_aa.long(), num_classes=q).to(dtype).to(device)
    params_e = { # only biases and couplings needed for energy computation 
        "bias": params["bias"],
        "coupling_matrix": params["coupling_matrix"]
    }
    energy_initial = compute_energy(initial_chains_onehot, params_e)
    energy_final = compute_energy(current_chains, params_e)
    
    mean_energy_initial = energy_initial.mean().item()
    mean_energy_final = energy_final.mean().item()
    energy_change = mean_energy_final - mean_energy_initial
    
    print(f"  Mean energy (initial): {mean_energy_initial:.4f}")
    print(f"  Mean energy (final):   {mean_energy_final:.4f}")
    print(f"  Energy change:         {energy_change:.4f}")
    print(f"  (took {time.time()-t0:.2f}s)")

    print("-" * 60)
    evolution_time = time.time() - t_evolution_start
    print(f"\nEvolution completed!")
    print(f"  Total iterations: {num_iterations}")
    print(f"  Evolution time: {evolution_time:.2f}s ({num_iterations/evolution_time:.1f} iter/s)")
    print(f"  Final chains: {current_chains.shape}")
    print(f"\nSampling method breakdown:")
    print(f"  Metropolis total time: {total_metropolis_time:.2f}s ({100*total_metropolis_time/evolution_time:.1f}%)")
    print(f"  Gibbs total time:      {total_gibbs_time:.2f}s ({100*total_gibbs_time/evolution_time:.1f}%)")
    print(f"  Final DNA chains: {current_dna_chains.shape}")
    print()
    
    # Save results to file
    print("Saving evolution results...")
    output_file = os.path.join(args.output, "evolution_results.txt")
    
    # Check for gap mutations (gap should NEVER mutate)
    gap_positions_initial = (initial_chains_aa == 0).cpu().numpy()
    gap_positions_final = (final_chains_aa == 0).cpu().numpy()
    gap_mutations = gap_positions_initial != gap_positions_final
    total_gap_mutations = gap_mutations.sum()
    
    if total_gap_mutations > 0:
        print(f"WARNING: Found {total_gap_mutations} gap mutations! Gaps should NEVER mutate.")
    else:
        print("✓ Gap integrity verified: No gaps mutated during evolution")
    
    with open(output_file, "w") as f:
        f.write("EVOLUTION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total sequences: {N}\n")
        f.write(f"Sequence length: {L}\n")
        f.write(f"Total iterations: {num_iterations}\n\n")
        
        # Energy information
        f.write("ENERGY ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean energy (initial): {mean_energy_initial:.6f}\n")
        f.write(f"Mean energy (final):   {mean_energy_final:.6f}\n")
        f.write(f"Energy change:         {energy_change:.6f}\n")
        if energy_change < 0:
            f.write(f"✓ Energy decreased (system stabilized)\n\n")
        else:
            f.write(f"⚠ Energy increased\n\n")
        
        # Correlation tracking
        if len(correlation_history) > 0:
            f.write("CORRELATION ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Initial Pearson correlation: {correlation_history[0]:.6f}\n")
            f.write(f"Final Pearson correlation:   {correlation_history[-1]:.6f}\n")
            f.write(f"Correlation change:          {correlation_history[-1] - correlation_history[0]:.6f}\n\n")
        
        # Gap mutation check
        f.write("GAP MUTATION CHECK:\n")
        f.write("-" * 80 + "\n")
        if total_gap_mutations > 0:
            f.write(f"WARNING: {total_gap_mutations} gap positions mutated!\n")
            f.write("Gaps should NEVER change during evolution.\n\n")
            # Find which sequences and positions
            for seq_idx in range(N):
                seq_gap_muts = gap_mutations[seq_idx]
                if seq_gap_muts.any():
                    positions = [i for i, val in enumerate(seq_gap_muts) if val]
                    f.write(f"  Sequence {seq_idx}: gaps mutated at positions {positions}\n")
            f.write("\n")
        else:
            f.write("✓ PASS: No gaps mutated during evolution.\n")
            f.write("All gap positions remained unchanged.\n\n")
        
        # Show first 5 sequences
        num_display = min(5, N)
        for seq_idx in range(num_display):
            f.write("=" * 80 + "\n")
            f.write(f"SEQUENCE {seq_idx + 1}\n")
            f.write("=" * 80 + "\n\n")
            
            # Get sequences
            initial_aa = initial_chains_aa[seq_idx].cpu().numpy()
            initial_dna = initial_dna_chains[seq_idx].cpu().numpy()
            final_aa = final_chains_aa[seq_idx].cpu().numpy()
            final_dna = final_dna_chains[seq_idx].cpu().numpy()
            
            # Initial state
            f.write("BEFORE EVOLUTION:\n")
            f.write("-" * 80 + "\n")
            f.write("Amino acids: ")
            f.write("".join(tokens[int(aa)] for aa in initial_aa))
            f.write("\n")
            
            f.write("Codons:      ")
            f.write(" ".join(sampling_data["all_codons"][int(c)] for c in initial_dna))
            f.write("\n\n")
            
            # Final state
            f.write("AFTER EVOLUTION:\n")
            f.write("-" * 80 + "\n")
            f.write("Amino acids: ")
            f.write("".join(tokens[int(aa)] for aa in final_aa))
            f.write("\n")
            
            f.write("Codons:      ")
            f.write(" ".join(sampling_data["all_codons"][int(c)] for c in final_dna))
            f.write("\n\n")
            
            # Count and detail changes
            aa_changed_positions = [i for i in range(len(initial_aa)) if initial_aa[i] != final_aa[i]]
            dna_changed_positions = [i for i in range(len(initial_dna)) if initial_dna[i] != final_dna[i]]
            
            f.write("SUMMARY OF CHANGES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total changes: {len(aa_changed_positions)} amino acids, {len(dna_changed_positions)} codons\n\n")
            
            # Detail amino acid changes
            if aa_changed_positions:
                f.write("AMINO ACID CHANGES:\n")
                for pos in aa_changed_positions:
                    initial_letter = tokens[int(initial_aa[pos])]
                    final_letter = tokens[int(final_aa[pos])]
                    f.write(f"  Position {pos:3d}: {initial_letter} (idx {initial_aa[pos]:2d}) -> {final_letter} (idx {final_aa[pos]:2d})\n")
                f.write("\n")
            else:
                f.write("No amino acid changes.\n\n")
            
            # Detail codon changes
            if dna_changed_positions:
                f.write("CODON CHANGES:\n")
                for pos in dna_changed_positions:
                    initial_codon = sampling_data["all_codons"][int(initial_dna[pos])]
                    final_codon = sampling_data["all_codons"][int(final_dna[pos])]
                    # Check if amino acid also changed
                    aa_also_changed = " (AA CHANGED)" if pos in aa_changed_positions else " (synonymous)"
                    f.write(f"  Position {pos:3d}: {initial_codon} (idx {initial_dna[pos]:2d}) -> {final_codon} (idx {final_dna[pos]:2d}){aa_also_changed}\n")
                f.write("\n")
            else:
                f.write("No codon changes.\n\n")
    
    print(f"Results saved to {output_file}")
    print()
    
    # Save correlation history
    if len(correlation_history) > 0:
        print("Saving correlation history...")
        correlation_file = os.path.join(args.output, "correlation_history.txt")
        with open(correlation_file, "w") as f:
            f.write("Iteration\tPearson_Correlation\n")
            for iter_num, pearson in zip(iteration_checkpoints, correlation_history):
                f.write(f"{iter_num}\t{pearson:.6f}\n")
        print(f"Correlation history saved to {correlation_file}")
        
        # Plot correlation evolution
        plt.figure(figsize=(10, 6))
        plt.plot(iteration_checkpoints, correlation_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Pearson Correlation', fontsize=12)
        plt.title('Evolution of Pearson Correlation with Target Statistics', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        correlation_plot_path = os.path.join(args.output, "correlation_evolution.png")
        plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation plot saved to {correlation_plot_path}")
        print()
    
    # PCA projection and visualization
    if pca is not None:
        print("Performing PCA analysis...")
        t0 = time.time()
        
        # Project natural sequences (use subsampled if available)
        if pca_subsample_indices is not None:
            pca_sequences_for_plot = pca_sequences_onehot[pca_subsample_indices]
            print(f"  Using {len(pca_subsample_indices)} subsampled natural sequences for plotting")
        else:
            pca_sequences_for_plot = pca_sequences_onehot
        
        natural_projection = project_sequences(pca_sequences_for_plot, pca)
        print(f"  Natural sequences projected: {natural_projection.shape}")
        
        # Project initial and final sequences
        initial_projection = project_sequences(initial_chains_onehot, pca)
        final_projection = project_sequences(current_chains, pca)
        
        print(f"  Initial sequences projected: {initial_projection.shape}")
        print(f"  Final sequences projected: {final_projection.shape}")
        
        # Create plots
        pca_plot_path = os.path.join(args.output, "pca_evolution.png")
        plot_pca_evolution(
            initial_projection,
            final_projection,
            pca_plot_path,
            title=f"PCA: Sequence Evolution ({num_iterations} iterations)",
            natural_proj=natural_projection
        )
        
        pca_density_plot_path = os.path.join(args.output, "pca_evolution_density.png")
        plot_pca_with_density(
            initial_projection,
            final_projection,
            pca_density_plot_path,
            title=f"PCA: Sequence Evolution ({num_iterations} iterations)",
            natural_proj=natural_projection
        )
        print(f"PCA analysis completed (took {time.time()-t0:.2f}s)")
        print()
    
    total_time = time.time() - start_time_total
    print("="*60)
    print(f"Genie 2.0 completed successfully")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("="*60)
    

if __name__ == "__main__":
    main()
