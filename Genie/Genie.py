"""
Genie 2.0 - Main Application
"""
import os
import sys
import torch
from adabmDCA import get_tokens, import_from_fasta
from .parser import parse_arguments


def main():
    """
    Main function for the Genie application.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if required input files exist
    if not os.path.isfile(args.path_sequences):
        print(f"Error: Sequences file not found: {args.path_sequences}", file=sys.stderr)
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
    print(f"  Sequences file    : {args.path_sequences}")
    print(f"  Parameters file   : {args.path_params}")
    print(f"  Output folder     : {args.output}")
    print(f"  Device            : {device}")
    print(f"  Data type         : {dtype}")
    print("-" * 60)
    print()
    
    # Set alphabet
    tokens = get_tokens("protein")
    print(f"Alphabet set to protein: {len(tokens)} tokens")
    
    # Load sequences from FASTA file
    print(f"Loading sequences from {args.path_sequences}...")
    _, sequences = import_from_fasta(args.path_sequences, tokens, filter_sequences=True)
    print(f"Loaded {len(sequences)} sequences")
    
    # Initialize application with parsed arguments
    # TODO: Add your initialization code here
    
    # Process arguments
    # TODO: Add your processing logic here
    
    # Execute main application logic
    # TODO: Add your main logic here
    
    print("Genie 2.0 started successfully")
    

if __name__ == "__main__":
    main()
