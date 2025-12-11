"""
Argument parser for Genie (codon-aware) application.
Handles command-line argument parsing for DNA sequence evolution with codon awareness.
"""
import argparse


def parse_arguments():
    """
    Parse command line arguments for the Genie application.
    
    Returns:
        argparse.Namespace: Parsed arguments with validated values
    """
    parser = argparse.ArgumentParser(
        description="Genie 2.0 - Codon-aware sequence evolution using DCA models with Metropolis-Hastings and Gibbs sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ============================================================================
    # REQUIRED ARGUMENTS
    # ============================================================================
    
    parser.add_argument(
        '-p', '--path_params',
        type=str,
        required=True,
        help='Path to the DCA parameters file (output from adabmDCA training)'
    )
    
    # ============================================================================
    # OPTIONAL INPUT ARGUMENTS
    # ============================================================================
    
    parser.add_argument(
        '-c', '--path_chains',
        type=str,
        default=None,
        help='Path to initial sequences file in FASTA format (if not provided, sequences are initialized randomly)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='DCA_evolution',
        help='Path to the output folder for results'
    )
    
    # ============================================================================
    # SEQUENCE INITIALIZATION OPTIONS
    # ============================================================================
    
    parser.add_argument(
        '--seq_index',
        type=int,
        default=None,
        help='Index of a single sequence to replicate N times (optional, for starting from one wild-type sequence)'
    )
    
    parser.add_argument(
        '-n', '--num_chains',
        type=int,
        default=None,
        help='Number of chains to evolve (required if --path_chains is not provided)'
    )
    
    # ============================================================================
    # SAMPLING PARAMETERS
    # ============================================================================
    
    parser.add_argument(
        '--p_metropolis',
        type=float,
        default=0.5,
        help='Probability threshold for Metropolis vs Gibbs sampling (0.0=only Metropolis for gaps, 1.0=only Gibbs, 0.5=50/50 mix)'
    )
    
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=50000,
        help='Number of evolution iterations (one mutation attempt per chain per iteration)'
    )
    
    parser.add_argument(
        '--save_steps',
        type=str,
        default='100',
        help='Save checkpoint every N iterations (int) or at specific iterations (comma-separated list, e.g., "100,500,1000"). Default: 100'
    )
    
    parser.add_argument(
        '--no-correlation-tracking',
        action='store_true',
        help='Disable correlation tracking during evolution (faster, avoids GPU-CPU data transfers)'
    )
    
    # ============================================================================
    # OPTIONAL DATA ANALYSIS
    # ============================================================================
    
    parser.add_argument(
        '-d', '--data',
        type=str,
        default=None,
        help='Path to reference dataset for PCA training (currently not used in main workflow)'
    )
    
    # ============================================================================
    # ARGUMENT VALIDATION
    # ============================================================================
    
    args = parser.parse_args()
    
    # Validate that num_chains is provided if path_chains is not
    if args.path_chains is None:
        if args.num_chains is None:
            parser.error("--num_chains is required when --path_chains is not provided")
    
    return args

