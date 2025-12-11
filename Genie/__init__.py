# Genie package
from .main import main
from .core import evolve_sequences
from .utils import (
    build_codon_neighbors,
    build_codon_to_index_map,
    build_amino_to_codons_map,
    parse_arguments
)

# Import reconstruction functions from scripts
from scripts import reconstruct_at_timesteps, reconstruct_chains_from_log

__version__ = "2.0.0"
__all__ = [
    "main",
    "evolve_sequences",
    "build_codon_neighbors",
    "build_codon_to_index_map",
    "build_amino_to_codons_map",
    "parse_arguments",
    "reconstruct_at_timesteps",
    "reconstruct_chains_from_log"
]


