"""
Argument parser for Genie application.
"""
import argparse


def parse_arguments():
    """
    Parse command line arguments for the Genie application.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Genie 2.0 - Description here",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-p', '--path_params',
        type=str,
        required=True,
        help='Path to the parameters file'
    )
    
    # Optional arguments
    parser.add_argument(
        '-c', '--path_chains',
        type=str,
        default=None,
        help='Path to the initial sequences file (optional, if not provided chains are initialized randomly)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='DCA_evolution',
        help='Path to the output folder (default: DCA_evolution)'
    )
    
    parser.add_argument(
        '--seq_index',
        type=int,
        default=None,
        help='Index of a single sequence to replicate N times (optional)'
    )
    
    parser.add_argument(
        '-n', '--num_chains',
        type=int,
        default=None,
        help='Number of chains to use during evolution (required if path_chains is not provided, otherwise optional)'
    )
    
    parser.add_argument(
        '--p_metropolis',
        type=float,
        default=0.5,
        help='Probability threshold for Metropolis vs Gibbs sampling (0.0 = only Metropolis, 1.0 = only Gibbs, default: 0.5)'
    )
    
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=50000,
        help='Number of evolution iterations (default: 50000)'
    )
    
    parser.add_argument(
        '--no-correlation-tracking',
        action='store_true',
        help='Disable correlation tracking during evolution (faster, less GPU-CPU transfers)'
    )
    
    parser.add_argument(
        '--no-pca',
        action='store_true',
        help='Disable PCA analysis (faster for large runs)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.path_chains is None:
        if args.num_chains is None:
            parser.error("--num_chains is required when --path_chains is not provided")
    
    return args
