"""
Codon utilities for genetic sequence manipulation.
"""

from itertools import product
import torch
from typing import Dict, List, Tuple


# Standard genetic code: codon to amino acid letter
CODON_TO_AMINO_LETTER = {
    "ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
    "AAC": "N", "AAT": "N", "AAA": "K", "AAG": "K",
    "AGC": "S", "AGT": "S", "AGA": "R", "AGG": "R",
    "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
    "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "CAC": "H", "CAT": "H", "CAA": "Q", "CAG": "Q",
    "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
    "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
    "GAC": "D", "GAT": "D", "GAA": "E", "GAG": "E",
    "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
    "TTC": "F", "TTT": "F", "TTA": "L", "TTG": "L",
    "TAC": "Y", "TAT": "Y", "TGC": "C", "TGT": "C", "TGG": "W",
    "---": "-"
}


def build_codon_to_index_map(tokens):
    """
    Build a mapping from codons to amino acid indices based on token alphabet.
    
    Args:
        tokens: String of amino acid tokens (e.g., '-ACDEFGHIKLMNPQRSTVWY' from adabmDCA)
    
    Returns:
        dict: Mapping {codon: index} where index is the position in tokens
    """
    # Create amino acid letter to index mapping
    aa_to_index = {aa: idx for idx, aa in enumerate(tokens)}
    
    # Map each codon to its amino acid index
    codon_to_index = {
        codon: aa_to_index[aa]
        for codon, aa in CODON_TO_AMINO_LETTER.items()
    }
    
    return codon_to_index


def build_amino_to_codons_map(codon_to_amino):
    """
    Build a mapping from amino acid indices to their corresponding codons.
    
    Args:
        codon_to_amino: Dictionary mapping codons to amino acid indices
    
    Returns:
        dict: Mapping {amino_index: [codons]} for indices 0-20
    """
    amino_to_codons = {}
    
    for amino in range(21):
        codons = [codon for codon, aa in codon_to_amino.items() if aa == amino]
        amino_to_codons[amino] = codons
    
    return amino_to_codons




def build_codon_neighbors():
    """
    Build a mapping of single-mutation nearest neighbor codons.
    
    For each non-stop codon, generates all accessible codons through a single
    nucleotide substitution at each position, excluding stop codons.
    
    Returns:
        tuple: A pair (neighbors_map, neighbor_counts) where:
            - neighbors_map: {codon: {position: [neighbor_codons]}}
            - neighbor_counts: {(codon, position): count}
              where position ∈ {0, 1, 2}
              
    Example:
        >>> neighbors, counts = build_codon_neighbors()
        >>> neighbors['ATG'][0]  # Nearest neighbors via mutation at position 0
        ['CTG', 'GTG', 'TTG']
        >>> counts[('ATG', 0)]  # Number of neighbors at position 0
        3
    """
    NUCLEOTIDES = ('A', 'C', 'G', 'T')
    STOP_CODONS = {'TAA', 'TAG', 'TGA'}
    
    neighbors_map = {}
    neighbor_counts = {}
    
    # Generate all possible codons using itertools.product for efficiency
    for codon_tuple in product(NUCLEOTIDES, repeat=3):
        codon = ''.join(codon_tuple)
        
        # Skip stop codons
        if codon in STOP_CODONS:
            continue
        
        # Build neighbor map for each position
        position_neighbors = {}
        for position in range(3):
            neighbors = [
                codon[:position] + nucleotide + codon[position + 1:]
                for nucleotide in NUCLEOTIDES
                if nucleotide != codon[position]
                and codon[:position] + nucleotide + codon[position + 1:] not in STOP_CODONS
            ]
            position_neighbors[position] = neighbors
            neighbor_counts[(codon, position)] = len(neighbors)
        
        neighbors_map[codon] = position_neighbors
    
    return neighbors_map, neighbor_counts


def build_codon_neighbor_tensor(
    codon_neighbors: Dict[str, Dict[int, List[str]]],
    codon_to_amino: Dict[str, int],
    num_amino_acids: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Build a GPU-friendly tensor representation of codon neighbor accessibility.
    
    Creates a 4D tensor where neighbor_tensor[codon_idx, position, aa_idx] indicates
    whether amino acid aa_idx is accessible from codon_idx by mutating nucleotide at position.
    
    Args:
        codon_neighbors: Dictionary mapping codons to their neighbors by position
        codon_to_amino: Dictionary mapping codons to amino acid indices
        num_amino_acids: Total number of amino acids (typically 21 including gap)
        device: Torch device for tensor allocation
    
    Returns:
        torch.Tensor: Boolean tensor of shape (num_codons, 3, num_amino_acids)
                     where [i, pos, aa] = True if aa is accessible from codon i at position pos
    """
    # Create codon to index mapping
    all_codons = sorted(codon_neighbors.keys())
    codon_to_idx = {codon: idx for idx, codon in enumerate(all_codons)}
    num_codons = len(all_codons)
    
    # Initialize tensor on specified device
    if device is None:
        device = torch.device('cpu')
    
    neighbor_tensor = torch.zeros(num_codons, 3, num_amino_acids, dtype=torch.bool, device=device)
    
    # Fill tensor
    for codon, codon_idx in codon_to_idx.items():
        for pos in range(3):
            # Get neighbor codons at this position
            neighbor_codons = codon_neighbors[codon].get(pos, [])
            
            # Mark accessible amino acids
            for neighbor_codon in neighbor_codons:
                if neighbor_codon in codon_to_amino:
                    aa_idx = codon_to_amino[neighbor_codon]
                    neighbor_tensor[codon_idx, pos, aa_idx] = True
            
            # Include current amino acid (no mutation)
            if codon in codon_to_amino:
                current_aa = codon_to_amino[codon]
                neighbor_tensor[codon_idx, pos, current_aa] = True
    
    return neighbor_tensor, codon_to_idx, all_codons


def build_codon_mutation_lookup(
    codon_neighbors: Dict[str, Dict[int, List[str]]],
    codon_to_amino: Dict[str, int],
    device: torch.device = None
) -> Tuple[torch.Tensor, Dict[str, int], List[str]]:
    """
    Build a complete lookup tensor for codon mutations.
    
    For each (codon, position, target_amino_acid), stores indices of valid neighbor codons.
    
    Args:
        codon_neighbors: Dictionary mapping codons to their neighbors by position
        codon_to_amino: Dictionary mapping codons to amino acid indices
        device: Torch device for tensor allocation
    
    Returns:
        Tuple of:
        - mutation_lookup: Tensor (num_codons, 3, num_aa, max_neighbors) with codon indices
        - num_options: Tensor (num_codons, 3, num_aa) with count of valid options
        - codon_to_idx: Dictionary mapping codon strings to indices
        - idx_to_codon: List mapping indices back to codon strings
    """
    # Create codon mappings
    all_codons = sorted(codon_neighbors.keys())
    codon_to_idx = {codon: idx for idx, codon in enumerate(all_codons)}
    num_codons = len(all_codons)
    num_aa = 21  # Including gap
    
    if device is None:
        device = torch.device('cpu')
    
    # Find maximum number of neighbor options for any (codon, pos, aa) combination
    max_neighbors = 0
    for codon in all_codons:
        for pos in range(3):
            neighbor_codons = codon_neighbors[codon].get(pos, [])
            # Group by amino acid
            aa_groups = {}
            for neighbor in neighbor_codons:
                if neighbor in codon_to_amino:
                    aa = codon_to_amino[neighbor]
                    if aa not in aa_groups:
                        aa_groups[aa] = []
                    aa_groups[aa].append(neighbor)
            
            # Also include current codon
            if codon in codon_to_amino:
                current_aa = codon_to_amino[codon]
                if current_aa not in aa_groups:
                    aa_groups[current_aa] = []
                if codon not in aa_groups[current_aa]:
                    aa_groups[current_aa].append(codon)
            
            for aa, codons in aa_groups.items():
                max_neighbors = max(max_neighbors, len(codons))
    
    # Initialize tensors
    mutation_lookup = torch.full((num_codons, 3, num_aa, max_neighbors), -1, dtype=torch.long, device=device)
    num_options = torch.zeros((num_codons, 3, num_aa), dtype=torch.long, device=device)
    
    # Fill lookup tensor
    for codon, codon_idx in codon_to_idx.items():
        for pos in range(3):
            neighbor_codons = codon_neighbors[codon].get(pos, [])
            
            # Group neighbors by amino acid
            aa_to_codons = {}
            for neighbor in neighbor_codons:
                if neighbor in codon_to_amino:
                    aa = codon_to_amino[neighbor]
                    if aa not in aa_to_codons:
                        aa_to_codons[aa] = []
                    aa_to_codons[aa].append(codon_to_idx[neighbor])
            
            # Include current codon
            if codon in codon_to_amino:
                current_aa = codon_to_amino[codon]
                if current_aa not in aa_to_codons:
                    aa_to_codons[current_aa] = []
                current_idx = codon_to_idx[codon]
                if current_idx not in aa_to_codons[current_aa]:
                    aa_to_codons[current_aa].append(current_idx)
            
            # Store in tensor
            for aa, codon_indices in aa_to_codons.items():
                count = len(codon_indices)
                num_options[codon_idx, pos, aa] = count
                mutation_lookup[codon_idx, pos, aa, :count] = torch.tensor(codon_indices, dtype=torch.long, device=device)
    
    return mutation_lookup, num_options, codon_to_idx, all_codons


def precompute_sampling_tensors(
    tokens: str,
    device: torch.device
) -> Dict:
    """
    Pre-compute all necessary tensors for GPU-optimized Gibbs sampling.
    
    This function builds all required data structures and tensors in one call,
    including codon mappings, neighbor networks, and lookup tables.
    
    Args:
        tokens: Amino acid alphabet string (e.g., from get_tokens("protein"))
        device: Torch device for tensor allocation (CPU/GPU)
    
    Returns:
        Dict containing:
            - "codon_to_amino": Dict mapping codons to amino acid indices
            - "amino_to_codons": Dict mapping amino acid indices to codon lists
            - "codon_neighbors": Dict of codon nearest neighbors by position
            - "neighbor_counts": Dict of neighbor counts
            - "codon_neighbor_tensor": Tensor (num_codons, 3, q) for accessibility
            - "mutation_lookup": Tensor (num_codons, 3, q, max_neighbors) for mutations
            - "num_options": Tensor (num_codons, 3, q) for option counts
            - "codon_to_idx": Dict mapping codon strings to indices
            - "all_codons": List of all codons
    """
    num_amino_acids = len(tokens)
    
    # Build codon mappings
    codon_to_amino = build_codon_to_index_map(tokens)
    amino_to_codons = build_amino_to_codons_map(codon_to_amino)
    
    # Build codon neighbor network
    codon_neighbors, neighbor_counts = build_codon_neighbors()
    
    # Build GPU tensors for fast sampling
    codon_neighbor_tensor, codon_to_idx, all_codons = build_codon_neighbor_tensor(
        codon_neighbors, 
        codon_to_amino, 
        num_amino_acids, 
        device
    )
    
    mutation_lookup, num_options, codon_to_idx, all_codons = build_codon_mutation_lookup(
        codon_neighbors,
        codon_to_amino,
        device
    )
    
    return {
        "codon_to_amino": codon_to_amino,
        "amino_to_codons": amino_to_codons,
        "codon_neighbors": codon_neighbors,
        "neighbor_counts": neighbor_counts,
        "codon_neighbor_tensor": codon_neighbor_tensor,
        "mutation_lookup": mutation_lookup,
        "num_options": num_options,
        "codon_to_idx": codon_to_idx,
        "all_codons": all_codons
    }
