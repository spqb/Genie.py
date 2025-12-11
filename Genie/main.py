"""
Genie 2.0 - DNA Codon Evolution with DCA Models

GPU-accelerated MCMC sampling for protein sequences with codon-level mutations.
Combines Direct Coupling Analysis (DCA) models with biologically realistic 
codon substitution dynamics.
"""
import os
import sys
import time

import torch
import numpy as np
import pandas as pd
from adabmDCA import get_tokens, import_from_fasta, load_params
from adabmDCA.fasta import write_fasta
from adabmDCA.dataset import DatasetDCA
from adabmDCA.stats import get_freq_single_point, get_freq_two_points

from .utils.parser import parse_arguments
from .utils.codon_utils import precompute_sampling_tensors
from .core.evolution import evolve_sequences
from .utils.stats import compute_target_statistics, compute_correlations

# ============================================================================
# PyTorch JIT Compilation Setup
# ============================================================================
# Compile the evolution function for maximum performance using torch.compile()
# Mode 'max-autotune': Extensive optimization, 2-3x faster (10-30s warmup on first call)
# fullgraph=False: Allows graph breaks for Python scalar operations
# dynamic=False: Assumes fixed tensor shapes for better optimization
try:
    evolve_sequences_compiled = torch.compile(
        evolve_sequences,
        mode='max-autotune',
        fullgraph=False,
        dynamic=False
    )
    COMPILE_AVAILABLE = True
except Exception as e:
    print(f"Warning: torch.compile() not available ({e}). Using uncompiled version.")
    evolve_sequences_compiled = evolve_sequences
    COMPILE_AVAILABLE = False

# ============================================================================
# Helper Functions
# ============================================================================

def translate_to_dna_uniform(
    chains: torch.Tensor,
    amino_to_codons_tensor: torch.Tensor,
    amino_to_num_codons: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Translate amino acid sequences to DNA codon sequences with uniform codon selection.
    
    This function performs GPU-optimized, fully vectorized translation. For each amino
    acid in the input sequences, it randomly selects one codon uniformly from all 
    available synonymous codons. No CPU loops are used - all operations are batched
    on the GPU for maximum performance.
    
    Args:
        chains: Amino acid sequences as indices, shape (N, L)
                where N = number of sequences, L = sequence length
        amino_to_codons_tensor: Mapping from amino acids to codon indices, 
                               shape (num_amino_acids, max_codons_per_aa)
        amino_to_num_codons: Number of codons available for each amino acid,
                            shape (num_amino_acids,)
        device: PyTorch device (CPU or CUDA)
    
    Returns:
        DNA codon sequences as indices, shape (N, L)
    
    Example:
        If position (i, j) has amino acid 'A' (alanine) with 4 possible codons,
        one of those 4 codons is selected uniformly at random.
    """
    N, L = chains.shape
    
    # Flatten sequences for vectorized operations (N*L positions total)
    chains_flat = chains.flatten()  # Shape: (N*L,)
    
    # Look up number of available codons for each amino acid position
    num_codons_per_position = amino_to_num_codons[chains_flat]  # Shape: (N*L,)
    
    # Generate random codon selection indices (uniform within available codons)
    # Random float [0, 1) * num_codons gives uniform selection
    random_codon_indices = torch.rand(N * L, device=device) * num_codons_per_position.float()
    random_codon_indices = random_codon_indices.long()  # Convert to integer indices
    
    # Select codons using advanced indexing
    # For each position i: amino_to_codons_tensor[chains_flat[i], random_codon_indices[i]]
    selected_codons = amino_to_codons_tensor[chains_flat, random_codon_indices]
    
    # Reshape back to original sequence dimensions
    dna_chains = selected_codons.reshape(N, L)
    
    return dna_chains


# ============================================================================
# Main Application
# ============================================================================

def main():
    """
    Main entry point for Genie 2.0 - DNA codon evolution with DCA models.
    
    This function orchestrates the complete workflow:
    1. Parse command-line arguments
    2. Initialize GPU/CPU device and load DCA model parameters  
    3. Load or generate initial protein sequences
    4. Translate sequences to DNA codons
    5. Run MCMC sampling with codon-aware mutations
    6. Track convergence metrics (optional, if reference data provided)
    7. Save results and statistics
    """
    start_time_total = time.time()
    
    # ========================================================================
    # Argument Parsing and Input Validation
    # ========================================================================
    args = parse_arguments()
    
    # ========================================================================
    # Redirect stdout and stderr to log file
    # ========================================================================
    folder = args.output
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, "genie.log")
    log_handle = open(log_file, "w")
    sys.stdout = log_handle
    sys.stderr = log_handle
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Validate input files exist
    if args.path_chains is not None and not os.path.isfile(args.path_chains):
        print(f"Error: Sequences file not found: {args.path_chains}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(args.path_params):
        print(f"Error: Parameters file not found: {args.path_params}", file=sys.stderr)
        sys.exit(1)
    
    # ========================================================================
    # Device Setup and Configuration
    # ========================================================================
    # Detect and configure GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {device_name}")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU")
    
    dtype = torch.float32  # Use 32-bit floats for GPU efficiency
    
    # ========================================================================
    # Configuration Display
    # ========================================================================
    num_iterations = args.num_iterations
    print("\n" + "="*80)
    print("  GENIE 2.0 - DNA Codon Evolution")
    print("="*80 + "\n")
    
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    
    # Display input configuration
    if args.path_chains is not None:
        print(template.format("Sequences file:", args.path_chains))
    else:
        print(template.format("Sequences:", "Random initialization"))
        print(template.format("Number of chains:", args.num_chains))
        print(template.format("Sequence length:", "auto-detect from DCA params"))
    
    if args.data is not None:
        print(template.format("Reference data:", args.data))
    
    # Display model and run parameters
    print(template.format("Parameters file:", args.path_params))
    print(template.format("Output folder:", args.output))
    print(template.format("Number of iterations:", f"{num_iterations:,}"))
    print(template.format("P (Metropolis):", f"{args.p_metropolis} (Metropolis/Gibbs ratio)"))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", str(dtype)))
    print("-" * 80 + "\n")
    
    # ========================================================================
    # Alphabet and Codon Network Initialization
    # ========================================================================
    print("[INITIALIZATION]")
    print("-" * 80)
    
    # Load protein alphabet (20 amino acids + gap symbol)
    t0 = time.time()
    tokens = get_tokens("protein")
    print(f"  Setting alphabet to protein: {len(tokens)} tokens")
    print(f"  ✓ Alphabet configured ({time.time()-t0:.2f}s)")
    
    # Pre-compute codon network and mutation lookup tables
    # This builds the graph of all possible single-nucleotide codon mutations
    # and maps them to the resulting amino acid changes
    print(f"  Pre-computing sampling tensors...")
    t0 = time.time()
    sampling_data = precompute_sampling_tensors(tokens, device)
    elapsed = time.time() - t0
    print(f"  ✓ Sampling tensors ready ({len(sampling_data['codon_neighbors'])} codons, {elapsed:.2f}s)")
    print("-" * 80 + "\n")
    
    # ========================================================================
    # DCA Model Loading
    # ========================================================================
    print("[MODEL LOADING]")
    print("-" * 80)
    print(f"  Loading parameters from: {args.path_params}")
    t0 = time.time()
    params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
    L_params = params["bias"].shape[0]  # Sequence length from bias field shape (L, q)
    q = len(tokens)  # Number of states (amino acids + gap)
    print(f"  ✓ Parameters loaded (L={L_params}, q={q}, {time.time()-t0:.2f}s)")
    
    # ========================================================================
    # Sequence Loading or Initialization
    # ========================================================================
    if args.path_chains is not None:
        # Load existing sequences from FASTA file
        print(f"  Loading sequences from file...")
        t0 = time.time()
        _, sequences = import_from_fasta(args.path_chains, tokens, filter_sequences=True)
        print(f"  ✓ Loaded {len(sequences)} sequences ({time.time()-t0:.2f}s)")
        
        # Ensure sequences are torch tensors on correct device
        if not isinstance(sequences, torch.Tensor):
            sequences = torch.tensor(sequences, dtype=torch.long)
        sequences = sequences.to(device)
        
        chains = sequences  # Shape: (N, L) where N=num_sequences, L=length
        N, L = chains.shape
        
        # Verify sequence length matches DCA model expectations
        if L != L_params:
            print(f"  ⚠ Warning: Loaded sequences have length {L}, but DCA model expects {L_params}")
    else:
        # Initialize random sequences with length from DCA model
        L = L_params
        N = args.num_chains
        print(f"  Initializing {N} random sequences of length {L}...")
        # Random initialization excluding gap (token 0)
        chains = torch.randint(1, len(tokens), (N, L), dtype=torch.long, device=device)
        print(f"  ✓ Random sequences initialized: {chains.shape}")
    
    # Integrate sampling tensors into params dict for easy access
    params.update(sampling_data)
    print(f"  ✓ Sampling tensors integrated")
    print("-" * 80 + "\n")
    
    # ========================================================================
    # Optional Sequence Selection/Replication
    # ========================================================================
    # If num_chains specified with loaded sequences, select random subset
    if args.num_chains is not None and args.path_chains is not None:
        if args.num_chains <= 0:
            print(f"Error: num_chains must be positive, got {args.num_chains}", file=sys.stderr)
            sys.exit(1)
        if args.num_chains > N:
            print(f"  ⚠ Warning: num_chains {args.num_chains} exceeds available {N}. Using all sequences.")
        else:
            print("[SEQUENCE SELECTION]")
            print("-" * 80)
            print(f"  Selecting {args.num_chains} random sequences from {N}...")
            indices = torch.randperm(N)[:args.num_chains]
            chains = chains[indices]
            N = args.num_chains
            print(f"  ✓ Using {N} sequences for evolution")
            print("-" * 80 + "\n")
    
    # If seq_index specified, replicate that single sequence N times
    # Useful for exploring variants of a specific sequence
    if args.seq_index is not None:
        if args.path_chains is None:
            print(f"Error: --seq_index requires --path_chains", file=sys.stderr)
            sys.exit(1)
        if args.seq_index < 0 or args.seq_index >= chains.shape[0]:
            print(f"Error: seq_index {args.seq_index} out of range [0, {chains.shape[0]-1}]", file=sys.stderr)
            sys.exit(1)
        
        print("[SEQUENCE REPLICATION]")
        print("-" * 80)
        print(f"  Replicating sequence {args.seq_index} {N} times...")
        selected_sequence = chains[args.seq_index:args.seq_index+1]  # Shape: (1, L)
        chains = selected_sequence.repeat(N, 1)  # Replicate to (N, L)
        print(f"  ✓ All {N} sequences are now identical to sequence {args.seq_index}")
        print("-" * 80 + "\n")
    

    # ========================================================================
    # DNA Codon Translation
    # ========================================================================
    print("[SEQUENCE PREPARATION]")
    print("-" * 80)
    print("  Translating amino acid sequences to DNA codons...")
    t0 = time.time()
    dna_chains = translate_to_dna_uniform(
        chains, 
        sampling_data["amino_to_codons_tensor"], 
        sampling_data["amino_to_num_codons"], 
        device
    )
    print(f"  ✓ DNA sequences generated: {dna_chains.shape} ({time.time()-t0:.2f}s)")

    # Convert amino acid indices to one-hot encoding for energy calculations
    # Shape: (N, L, q) where q is number of amino acid states
    chains_onehot = torch.nn.functional.one_hot(chains.long(), num_classes=q).to(dtype).to(device)
    print(f"  ✓ One-hot encoding: {chains_onehot.shape}")
    print("-" * 80 + "\n")
    
    # ========================================================================
    # MCMC Sampling Setup
    # ========================================================================
    # Initialize current state (will be mutated during sampling)
    current_chains = chains_onehot.clone()
    current_dna_chains = dna_chains.clone()
    
    # Pre-generate random numbers in batches for GPU efficiency
    # Batch size of 1000 reduces kernel launch overhead
    random_batch_size = min(1000, num_iterations)
    random_batch = torch.rand(random_batch_size, N, device=device)
    random_batch_idx = 0

    # Use JIT-compiled version if available for 2-3x speedup
    evolve_fn = evolve_sequences_compiled if COMPILE_AVAILABLE else evolve_sequences
    
    # ========================================================================
    # Parse save_steps: can be int (periodic) or list (specific iterations)
    # ========================================================================
    if ',' in args.save_steps:
        # Comma-separated list of specific iterations
        save_steps_list = sorted([int(x.strip()) for x in args.save_steps.split(',')])
        save_steps_set = set(save_steps_list)
        save_mode = "list"
    else:
        # Single integer for periodic checkpoints
        save_steps_period = int(args.save_steps)
        save_mode = "periodic"




    # ========================================================================
    # MCMC Sampling with Optional Reference Data Tracking
    # ========================================================================
    if args.data is not None:
        # ====================================================================
        # Mode: Convergence Tracking Against Reference Data
        # ====================================================================
        print("[DATA ANALYSIS]")
        print("-" * 80)
        print(f"  Loading reference data from: {args.data}")
        
        # Load reference dataset for convergence monitoring
        clustering_seqid = 0.8  # Sequence identity threshold for clustering
        no_reweighting = False  # Use sequence weights
        dataset = DatasetDCA(
            path_data=args.data,
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

        print(f"  ✓ Data loaded ({len(dataset)} sequences)")
        pseudocount = 1. / dataset.weights.sum().item()
        print(f"  ✓ Pseudocount: {pseudocount:.6f}")
        print("-" * 80 + "\n")
        
        print("[MCMC SAMPLING]")
        print("-" * 80)
        print(f"  Total iterations: {num_iterations:,}")
        if COMPILE_AVAILABLE:
            print(f"  Compiler: torch.compile (mode='max-autotune')")
            print(f"  ⚠ First iteration slower due to compilation (~10-30s)")
        else:
            print(f"  Compiler: Not available (using eager mode)")
        print(f"  Starting MCMC sampling...")

        # Compute target statistics from reference data
        # fi: single-site frequencies, fij: pairwise frequencies
        fi = get_freq_single_point(data=dataset.data, weights=dataset.weights, pseudo_count=pseudocount)
        fij = get_freq_two_points(data=dataset.data, weights=dataset.weights, pseudo_count=pseudocount)

        # Storage for convergence metrics
        results_sampling = {
            "iteration" : [],
            "pearson" : [],
        }

        # ====================================================================
        # Save initial chains before MCMC
        # ====================================================================
        initial_chains_file = os.path.join(folder, "initial_chains.fasta")
        # Convert one-hot to indices then to amino acid letters
        initial_chains_idx = current_chains.argmax(dim=-1)  # Shape: (N, L)
        initial_headers = []
        initial_sequences = []
        for i in range(initial_chains_idx.shape[0]):
            seq = "".join([tokens[idx.item()] for idx in initial_chains_idx[i]])
            initial_headers.append(f"chain_{i}")
            initial_sequences.append(seq)
        write_fasta(initial_chains_file, initial_headers, initial_sequences)
        print(f"  ✓ Initial chains saved: {initial_chains_file}")

        # Open mutation log file for streaming writes (checkpoint-based)
        mutation_log_file = os.path.join(folder, "mutation_log.csv")
        mutation_log_handle = open(mutation_log_file, "w")
        mutation_log_handle.write("iteration,chain_id,position,new_aa\n")
        print(f"  ✓ Mutation log opened: {mutation_log_file}")
        if save_mode == "periodic":
            print(f"  ✓ Checkpoint interval: every {save_steps_period} iterations")
        else:
            print(f"  ✓ Checkpoint iterations: {save_steps_list}")

        # Store previous checkpoint chains for comparison
        prev_checkpoint_chains = current_chains.clone()

        # Main MCMC loop with convergence tracking
        t_evolution_start = time.time()
        for iteration in range(num_iterations):
            # Get pre-generated random values for Metropolis/Gibbs split
            p_values = random_batch[random_batch_idx]
            random_batch_idx += 1
            
            # Regenerate batch when exhausted
            if random_batch_idx >= random_batch_size and iteration + 1 < num_iterations:
                remaining = num_iterations - (iteration + 1)
                random_batch_size = min(1000, remaining)
                random_batch = torch.rand(random_batch_size, N, device=device)
                random_batch_idx = 0
            
            # Evolution step: mutate all sequences in parallel
            current_chains, current_dna_chains, mutation_info_tensor = evolve_fn(
                chains=current_chains,
                dna_chains=current_dna_chains,
                params=params,
                codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
                codon_neighbor_codon_tensor=sampling_data["codon_neighbor_codon_tensor"],
                mutation_lookup=sampling_data["mutation_lookup"],
                num_options=sampling_data["num_options"],
                codon_usage=sampling_data["codon_usage"],
                p=args.p_metropolis,
                p_values=p_values,
                device=device,
                dtype=dtype,
                beta=1.0
            )

            # Checkpoint-based mutation tracking: write mutations every save_steps iterations
            if save_mode == "periodic":
                is_checkpoint = (iteration + 1) % save_steps_period == 0
            else:
                is_checkpoint = (iteration + 1) in save_steps_set
            is_last_iteration = (iteration + 1) == num_iterations
            
            if is_checkpoint or is_last_iteration:
                # Compare current chains with previous checkpoint
                current_chains_idx = current_chains.argmax(dim=-1)  # Shape: (N, L)
                prev_chains_idx = prev_checkpoint_chains.argmax(dim=-1)
                
                # Find all positions that changed since last checkpoint
                diff_mask = (current_chains_idx != prev_chains_idx)
                
                # Write mutations to file (only changed positions)
                for chain_id in range(N):
                    changed_positions = torch.where(diff_mask[chain_id])[0]
                    for pos in changed_positions:
                        pos_val = pos.item()
                        new_aa_idx = current_chains_idx[chain_id, pos_val].item()
                        new_aa_letter = tokens[new_aa_idx]
                        mutation_log_handle.write(f"{iteration},{chain_id},{pos_val},{new_aa_letter}\n")
                
                # Update checkpoint
                prev_checkpoint_chains = current_chains.clone()
                mutation_log_handle.flush()

            # Periodic convergence monitoring
            if (iteration + 1) % 1_000 == 0:
                elapsed = time.time() - t_evolution_start
                iter_per_sec = (iteration + 1) / elapsed
                print(f"  Iteration {iteration + 1:>7,}/{num_iterations:,} | {iter_per_sec:>6.1f} iter/s", end="")
                
                # Compute Pearson correlation between model and target
                pearson = compute_correlations(fi, fij, current_chains)
                
                # Track gap frequency agreement (amino acid position-specific)
                gap_freq_target = fi[:, 0].cpu()  # Reference gap frequencies
                pi_chains, _ = compute_target_statistics(current_chains)
                gap_freq_current = pi_chains[:, 0].cpu()
                gap_freq_diff = np.abs(gap_freq_target - gap_freq_current)
                max_gap_diff = gap_freq_diff.max()
                mean_gap_diff = gap_freq_diff.mean()
                pearson_gap = np.corrcoef(gap_freq_target, gap_freq_current)[0,1]
                
                print(f" | Pearson={pearson:.4f} | Gap: max={max_gap_diff:.4f}, avg={mean_gap_diff:.4f}, r={pearson_gap:.4f}")
    
                results_sampling["iteration"].append(iteration)
                results_sampling["pearson"].append(pearson)

            # First iteration timing (includes JIT compilation overhead)
            if iteration == 0:
                first_evolve_time = time.time() - t_evolution_start
                print(f"  ✓ First iteration completed ({first_evolve_time:.2f}s)")
            
            # Flush periodically to ensure data is written
            if (iteration + 1) % 1_000 == 0:
                mutation_log_handle.flush()

        # Close mutation log file
        mutation_log_handle.close()
        print(f"  ✓ Mutation log saved: {mutation_log_file}")

        # Save final chains
        final_chains_file = os.path.join(folder, "final_chains.fasta")
        final_chains_idx = current_chains.argmax(dim=-1)  # Shape: (N, L)
        final_headers = []
        final_sequences = []
        for i in range(final_chains_idx.shape[0]):
            seq = "".join([tokens[idx.item()] for idx in final_chains_idx[i]])
            final_headers.append(f"chain_{i}")
            final_sequences.append(seq)
        write_fasta(final_chains_file, final_headers, final_sequences)
        print(f"  ✓ Final chains saved: {final_chains_file}")

        print(f"  ✓ Sampling completed")
        print(f"  Final Pearson correlation: {pearson:.4f}")
        print("-" * 80 + "\n")
    
    else:
        # ====================================================================
        # Mode: Standard Sampling (No Reference Data)
        # ====================================================================
        print("[MCMC SAMPLING]")
        print("-" * 80)
        print(f"  Total iterations: {num_iterations:,}")
        if COMPILE_AVAILABLE:
            print(f"  Compiler: torch.compile (mode='max-autotune')")
            print(f"  ⚠ First iteration slower due to compilation (~10-30s)")
        else:
            print(f"  Compiler: Not available (using eager mode)")
        print(f"  Starting MCMC sampling...")

        # ====================================================================
        # Save initial chains before MCMC
        # ====================================================================
        initial_chains_file = os.path.join(folder, "initial_chains.fasta")
        # Convert one-hot to indices then to amino acid letters
        initial_chains_idx = current_chains.argmax(dim=-1)  # Shape: (N, L)
        initial_headers = []
        initial_sequences = []
        for i in range(initial_chains_idx.shape[0]):
            seq = "".join([tokens[idx.item()] for idx in initial_chains_idx[i]])
            initial_headers.append(f"chain_{i}")
            initial_sequences.append(seq)
        write_fasta(initial_chains_file, initial_headers, initial_sequences)
        print(f"  ✓ Initial chains saved: {initial_chains_file}")

        # Open mutation log file for streaming writes (checkpoint-based)
        mutation_log_file = os.path.join(folder, "mutation_log.csv")
        mutation_log_handle = open(mutation_log_file, "w")
        mutation_log_handle.write("iteration,chain_id,position,new_aa\n")
        print(f"  ✓ Mutation log opened: {mutation_log_file}")
        if save_mode == "periodic":
            print(f"  ✓ Checkpoint interval: every {save_steps_period} iterations")
        else:
            print(f"  ✓ Checkpoint iterations: {save_steps_list}")

        # Store previous checkpoint chains for comparison
        prev_checkpoint_chains = current_chains.clone()

        t_evolution_start = time.time()
        
        # Main MCMC loop without convergence tracking
        for iteration in range(num_iterations):
            # Get pre-generated random values
            p_values = random_batch[random_batch_idx]
            random_batch_idx += 1
            
            # Regenerate batch when exhausted
            if random_batch_idx >= random_batch_size and iteration + 1 < num_iterations:
                remaining = num_iterations - (iteration + 1)
                random_batch_size = min(1000, remaining)
                random_batch = torch.rand(random_batch_size, N, device=device)
                random_batch_idx = 0
            
            # Evolution step
            current_chains, current_dna_chains, mutation_info_tensor = evolve_fn(
                chains=current_chains,
                dna_chains=current_dna_chains,
                params=params,
                codon_neighbor_tensor=sampling_data["codon_neighbor_tensor"],
                codon_neighbor_codon_tensor=sampling_data["codon_neighbor_codon_tensor"],
                mutation_lookup=sampling_data["mutation_lookup"],
                num_options=sampling_data["num_options"],
                codon_usage=sampling_data["codon_usage"],
                p=args.p_metropolis,
                p_values=p_values,
                device=device,
                dtype=dtype,
                beta=1.0
            )

            # Checkpoint-based mutation tracking: write mutations every save_steps iterations
            if save_mode == "periodic":
                is_checkpoint = (iteration + 1) % save_steps_period == 0
            else:
                is_checkpoint = (iteration + 1) in save_steps_set
            is_last_iteration = (iteration + 1) == num_iterations
            
            if is_checkpoint or is_last_iteration:
                # Compare current chains with previous checkpoint
                current_chains_idx = current_chains.argmax(dim=-1)  # Shape: (N, L)
                prev_chains_idx = prev_checkpoint_chains.argmax(dim=-1)
                
                # Find all positions that changed since last checkpoint
                diff_mask = (current_chains_idx != prev_chains_idx)
                
                # Write mutations to file (only changed positions)
                for chain_id in range(N):
                    changed_positions = torch.where(diff_mask[chain_id])[0]
                    for pos in changed_positions:
                        pos_val = pos.item()
                        new_aa_idx = current_chains_idx[chain_id, pos_val].item()
                        new_aa_letter = tokens[new_aa_idx]
                        mutation_log_handle.write(f"{iteration},{chain_id},{pos_val},{new_aa_letter}\n")
                
                # Update checkpoint
                prev_checkpoint_chains = current_chains.clone()
                mutation_log_handle.flush()

            # Progress reporting every 1000 iterations
            if (iteration + 1) % 1_000 == 0:
                elapsed = time.time() - t_evolution_start
                iter_per_sec = (iteration + 1) / elapsed
                print(f"  Iteration {iteration + 1:>7,}/{num_iterations:,} | {iter_per_sec:>6.1f} iter/s")
    
            # First iteration timing
            if iteration == 0:
                first_evolve_time = time.time() - t_evolution_start
                print(f"  ✓ First iteration completed ({first_evolve_time:.2f}s)")
        
        # Close mutation log file
        mutation_log_handle.close()
        print(f"  ✓ Mutation log saved: {mutation_log_file}")

        # Save final chains
        final_chains_file = os.path.join(folder, "final_chains.fasta")
        final_chains_idx = current_chains.argmax(dim=-1)  # Shape: (N, L)
        final_headers = []
        final_sequences = []
        for i in range(final_chains_idx.shape[0]):
            seq = "".join([tokens[idx.item()] for idx in final_chains_idx[i]])
            final_headers.append(f"chain_{i}")
            final_sequences.append(seq)
        write_fasta(final_chains_file, final_headers, final_sequences)
        print(f"  ✓ Final chains saved: {final_chains_file}")

        print(f"  ✓ Sampling completed")
        print("-" * 80 + "\n")
        results_sampling = {}  # No convergence tracking in this mode


    # ========================================================================
    # Results Output
    # ========================================================================
    print("[RESULTS]")
    print("-" * 80)
    
    # Save convergence log if reference data was provided
    if results_sampling:
        df_samp_log = pd.DataFrame.from_dict(results_sampling)
        samp_log_file = os.path.join(folder, "sampling.log")
        df_samp_log.to_csv(samp_log_file, index=False)
        print(f"  ✓ Sampling log: {samp_log_file}")
    
    # Display total execution time
    total_time = time.time() - start_time_total
    print(f"  ✓ Total execution time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print("-" * 80 + "\n")
    
    print("=" * 80)
    print("  SAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Output folder: {str(folder)}")
    print(f"  Log file: {log_file}")
    print("\n" + "=" * 80 + "\n")
    
    # Close log file
    log_handle.close()
    

if __name__ == "__main__":
    main()
