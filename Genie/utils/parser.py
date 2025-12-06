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
        '-s', '--path_sequences',
        type=str,
        required=True,
        help='Path to the initial sequences file'
    )
    
    parser.add_argument(
        '-p', '--path_params',
        type=str,
        required=True,
        help='Path to the parameters file'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='DCA_evolution',
        help='Path to the output folder (default: DCA_evolution)'
    )
    
    args = parser.parse_args()
    return args
