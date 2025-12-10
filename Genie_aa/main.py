"""
Genie-AA 2.0 - Amino Acid Evolution with DCA Models

GPU-accelerated Gibbs sampling for protein sequences at the amino acid level.
Simpler and faster than codon-aware evolution, suitable when DNA-level 
realism is not required.
"""
import argparse
import os
import sys
import pandas as pd
import torch
import time

from adabmDCA.fasta import get_tokens, write_fasta
from adabmDCA.io import load_params
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import init_chains, resample_sequences, get_device, get_dtype
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points

from .sampling import get_sampler
from .utils.parser import parse_arguments


# ============================================================================
# Main Application
# ============================================================================

def main():       
    """
    Main entry point for Genie-AA 2.0 - amino acid evolution with DCA models.
    
    This function orchestrates the complete workflow:
    1. Parse command-line arguments
    2. Initialize GPU/CPU device and load DCA model parameters
    3. Load or generate initial protein sequences
    4. Run MCMC Gibbs sampling at amino acid level
    5. Track convergence metrics (optional, if reference data provided)
    6. Save results and statistics
    """
    start_time_total = time.time()

    # ========================================================================
    # Argument Parsing
    # ========================================================================
    args = parse_arguments()

    # ========================================================================
    # Configuration Display
    # ========================================================================
    print("\n" + "="*80)
    print("  GENIE-AA 2.0 - Amino Acid Evolution")
    print("="*80 + "\n")

    beta = 1  # Inverse temperature (fixed at 1.0 for standard sampling)
    
    # Create output directory
    folder = args.output
    os.makedirs(folder, exist_ok=True)

    # ========================================================================
    # Device and Alphabet Setup
    # ========================================================================
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    
    print("[CONFIGURATION]")
    print("-" * 80)
    template = "  {0:<28} {1:<50}"
    print(template.format("Parameters file:", args.path_params))
    if args.data is not None:
        print(template.format("Reference data:", args.data))
    print(template.format("Output folder:", str(folder)))
    print(template.format("Number of samples:", args.num_chains))
    print(template.format("Number of iterations:", f"{args.num_iterations:,}"))
    print(template.format("Beta (temperature):", beta))
    print(template.format("Device:", str(device)))
    print(template.format("Data type:", args.dtype))
    print("-" * 80 + "\n")
    
    # ========================================================================
    # Input Validation
    # ========================================================================
    if args.data is not None and not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    if not os.path.exists(args.path_params):
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")
        
    # ========================================================================
    # DCA Model Loading
    # ========================================================================
    print("[MODEL LOADING]")
    print("-" * 80)
    print(f"  Loading parameters from: {args.path_params}")
    params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
    L, q = params["bias"].shape  # L = sequence length, q = number of states
    print(f"  ✓ Parameters loaded (L={L}, q={q})")
    
    # Initialize Gibbs sampler (JIT-compiled for performance)
    print(f"  Initializing Gibbs sampler...")
    sampler = torch.jit.script(get_sampler("gibbs"))
    print(f"  ✓ Sampler ready")
    
    # ========================================================================
    # Sequence Loading or Initialization
    # ========================================================================
    if args.path_chains is not None:
        # Load existing sequences from FASTA file
        print(f"  Loading sequences from file...")
        t0 = time.time()
        from adabmDCA.fasta import import_from_fasta
        _, samples = import_from_fasta(args.path_chains, tokens, filter_sequences=True)
        print(f"  ✓ Loaded {len(samples)} sequences ({time.time()-t0:.2f}s)")
        
        # Ensure sequences are torch tensors on correct device
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples, dtype=torch.long)
        N_chains, L_chains = samples.shape

        # Verify sequence length matches DCA model
        if L_chains != L:
            print(f"  ⚠ Warning: Loaded sequences have length {L_chains}, but DCA model expects {L}")
        
        # Optional: select subset of sequences
        if args.num_chains is not None:
            if args.num_chains <= 0:
                print(f"Error: num_chains must be positive, got {args.num_chains}", file=sys.stderr)
                sys.exit(1)
            if args.num_chains > N_chains:
                print(f"  ⚠ Warning: num_chains {args.num_chains} exceeds available {N_chains}. Using all sequences.")
            else:
                print(f"  Selecting {args.num_chains} random sequences from {N_chains}...")
                indices = torch.randperm(N_chains)[:args.num_chains]
                samples = samples[indices]
                N_chains = args.num_chains
                print(f"  ✓ Using {N_chains} sequences")
            
        samples = samples.to(device)  # Move to GPU/CPU

    else:
        # Initialize random sequences
        N_chains = args.num_chains
        print(f"  Initializing {N_chains} random chains...")
        samples = init_chains(
            num_chains=N_chains,
            L=L,
            q=q,
            device=device,
            dtype=dtype,
        )
        print(f"  ✓ Chains initialized")
    
    print("-" * 80 + "\n")
    
    # ========================================================================
    # Optional Sequence Replication
    # ========================================================================
    # If seq_index specified, replicate that single sequence N_chains times
    if args.seq_index is not None:
        if args.path_chains is None:
            print(f"Error: --seq_index requires --path_chains", file=sys.stderr)
            sys.exit(1)
        if args.seq_index < 0 or args.seq_index >= samples.shape[0]:
            print(f"Error: seq_index {args.seq_index} out of range [0, {samples.shape[0]-1}]", file=sys.stderr)
            sys.exit(1)
        
        print("[SEQUENCE REPLICATION]")
        print("-" * 80)
        print(f"  Replicating sequence {args.seq_index} {N_chains} times...")
        selected_sequence = samples[args.seq_index:args.seq_index+1]  # Shape: (1, L)
        samples = selected_sequence.repeat(N_chains, 1)  # Replicate to (N_chains, L)
        print(f"  ✓ All {N_chains} sequences are now identical to sequence {args.seq_index}")
        print("-" * 80 + "\n")
        
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
        
        num_iterations = args.num_iterations
        print("[MCMC SAMPLING]")
        print("-" * 80)
        print(f"  Total iterations: {num_iterations:,}")
        print(f"  Sampler: Gibbs (JIT-compiled)")
        print(f"  Starting MCMC sampling...")

        # Compute target statistics from reference data
        fi = get_freq_single_point(data=dataset.data, weights=dataset.weights, pseudo_count=pseudocount)
        fij = get_freq_two_points(data=dataset.data, weights=dataset.weights, pseudo_count=pseudocount)

        # Storage for convergence metrics
        results_sampling = {
            "iteration" : [],
            "pearson" : [],
        }

        # Main MCMC loop with convergence tracking
        t_evolution_start = time.time()
        for iteration in range(num_iterations):
            # Gibbs sampling step: update all positions
            samples, mutation_info_tensor = sampler(chains=samples, params=params, beta=beta)
            
            # Periodic convergence monitoring
            if (iteration + 1) % 1_000 == 0:
                elapsed = time.time() - t_evolution_start
                iter_per_sec = (iteration + 1) / elapsed
                print(f"  Iteration {iteration + 1:>7,}/{num_iterations:,} | {iter_per_sec:>6.1f} iter/s", end="")
                
                # Compute statistics and Pearson correlation
                pi = get_freq_single_point(data=samples, weights=None, pseudo_count=0.)
                pij = get_freq_two_points(data=samples, weights=None, pseudo_count=0.)
                pearson, _ = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)
                results_sampling["iteration"].append(iteration)
                results_sampling["pearson"].append(pearson)
                print(f" | Pearson={pearson:.4f}")

            # First iteration timing
            if iteration == 0:
                first_time = time.time() - t_evolution_start
                print(f"  ✓ First iteration completed ({first_time:.2f}s)")

        print(f"  ✓ Sampling completed")
        print(f"  Final Pearson correlation: {pearson:.4f}")
        print("-" * 80 + "\n")
    
    else:
        # ====================================================================
        # Mode: Standard Sampling (No Reference Data)
        # ====================================================================
        num_iterations = args.num_iterations
        print("[MCMC SAMPLING]")
        print("-" * 80)
        print(f"  Total iterations: {num_iterations:,}")
        print(f"  Sampler: Gibbs (JIT-compiled)")
        print(f"  Starting MCMC sampling...")
        
        t_evolution_start = time.time()
        for iteration in range(num_iterations):
            # Gibbs sampling step
            samples, mutation_info_tensor = sampler(chains=samples, params=params, beta=beta)
            
            # Progress reporting every 1000 iterations
            if (iteration + 1) % 1_000 == 0:
                elapsed = time.time() - t_evolution_start
                iter_per_sec = (iteration + 1) / elapsed
                print(f"  Iteration {iteration + 1:>7,}/{num_iterations:,} | {iter_per_sec:>6.1f} iter/s")
            
            # First iteration timing
            if iteration == 0:
                first_time = time.time() - t_evolution_start
                print(f"  ✓ First iteration completed ({first_time:.2f}s)")

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
    print("\n" + "=" * 80 + "\n")
    
    
if __name__ == "__main__":
    main()