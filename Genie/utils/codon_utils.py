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
    aa_to_index = {aa: idx for idx, aa in enumerate(tokens)} # e.g., {'-':0, 'A':1, 'C':2, ...}
    
    # Map each codon to its amino acid index
    codon_to_index = {
        codon: aa_to_index[aa]
        for codon, aa in CODON_TO_AMINO_LETTER.items() 
    } # e.g., {'ATA':9, 'ATC':9, 'ATT':9, 'ATG':12, ...}
    
    return codon_to_index


def build_amino_to_codons_map(codon_to_amino):
    """
    Build a mapping from amino acid indices to their corresponding codons.
    
    Args:
        codon_to_amino: Dictionary mapping codons to amino acid indices (e.g., {'ATA':9, 'ATC':9, 'ATT':9, 'ATG':12, ...})
    
    Returns:
        dict: Mapping {amino_index: [codons]} for indices 0-20 (e.g., 9: ['ATA', 'ATC', 'ATT'], 12: ['ATG'], ...)
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
        codon = ''.join(codon_tuple)  # e.g., ('A','T','G') -> 'ATG'
        
        # Skip stop codons
        if codon in STOP_CODONS: 
            continue
        
        # Build neighbor map for each position
        position_neighbors = {}
        for position in range(3):
            neighbors = [
                codon[:position] + nucleotide + codon[position + 1:] # e.g., mutate 'ATG' at pos 0 with 'C' -> 'CTG'
                for nucleotide in NUCLEOTIDES
                if nucleotide != codon[position]
                and codon[:position] + nucleotide + codon[position + 1:] not in STOP_CODONS
            ] # e.g., for 'ATG' at pos 0: ['CTG', 'GTG', 'TTG']
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
    Also creates a codon_neighbor_codon_tensor[codon_idx, position, neighbor_codon_idx] indicating
    whether neighbor_codon_idx is accessible from codon_idx by mutating nucleotide at position.
    
    Args:
        codon_neighbors: Dictionary mapping codons to their neighbors by position
        codon_to_amino: Dictionary mapping codons to amino acid indices
        num_amino_acids: Total number of amino acids (typically 21 including gap)
        device: Torch device for tensor allocation
    
    Returns:
        tuple: (neighbor_tensor, codon_neighbor_codon_tensor, codon_to_idx, all_codons) where:
            - neighbor_tensor: Boolean tensor of shape (num_codons, 3, num_amino_acids)
                             where [i, pos, aa] = True if aa is accessible from codon i at position pos
            - codon_neighbor_codon_tensor: Boolean tensor of shape (num_codons, 3, num_codons)
                             where [i, pos, j] = True if codon j is accessible from codon i at position pos
            - codon_to_idx: Dictionary mapping codon strings to indices
            - all_codons: List of all codon strings
    """
    # Create codon to index mapping - include gap '---' and stop codons even though they have no neighbors
    STOP_CODONS = ['TAA', 'TAG', 'TGA']  # 3 stop codons
    all_codons = sorted(codon_neighbors.keys())
    
    # Add gap codon if not present (it won't be in codon_neighbors since it can't mutate)
    if '---' not in all_codons and '---' in codon_to_amino:
        all_codons = ['---'] + all_codons  # Add gap at the beginning
    
    # Add stop codons at the end (they can be proposed but immediately rejected)
    # Total: 1 gap + 61 coding + 3 stop = 65 codons
    for stop_codon in sorted(STOP_CODONS):
        if stop_codon not in all_codons:
            all_codons.append(stop_codon)
    
    codon_to_idx = {codon: idx for idx, codon in enumerate(all_codons)}
    num_codons = len(all_codons)
    
    # Initialize tensor on specified device
    if device is None:
        device = torch.device('cpu')
    
    neighbor_tensor = torch.zeros(num_codons, 3, num_amino_acids, dtype=torch.bool, device=device)
    codon_neighbor_codon_tensor = torch.zeros(num_codons, 3, num_codons, dtype=torch.bool, device=device)
    
    # Fill tensors
    for codon, codon_idx in codon_to_idx.items():
        # Special handling for gap codon - it can only stay as gap (no mutation)
        if codon not in codon_neighbors:
            if codon in codon_to_amino:
                gap_aa = codon_to_amino[codon]
                # For all positions, gap can only stay as gap
                for pos in range(3):
                    neighbor_tensor[codon_idx, pos, gap_aa] = True
                    codon_neighbor_codon_tensor[codon_idx, pos, codon_idx] = True
            continue
        
        for pos in range(3):
            # Get neighbor codons at this position
            neighbor_codons = codon_neighbors[codon].get(pos, [])
            
            # Mark accessible amino acids and codons
            for neighbor_codon in neighbor_codons:
                if neighbor_codon in codon_to_amino:
                    aa_idx = codon_to_amino[neighbor_codon]
                    neighbor_tensor[codon_idx, pos, aa_idx] = True
                    # Mark accessible codon
                    neighbor_codon_idx = codon_to_idx[neighbor_codon]
                    codon_neighbor_codon_tensor[codon_idx, pos, neighbor_codon_idx] = True
            
            # Include current amino acid and codon (no mutation)
            if codon in codon_to_amino:
                current_aa = codon_to_amino[codon]
                neighbor_tensor[codon_idx, pos, current_aa] = True
                codon_neighbor_codon_tensor[codon_idx, pos, codon_idx] = True
    
    return neighbor_tensor, codon_neighbor_codon_tensor, codon_to_idx, all_codons


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
    # Create codon mappings - include gap '---' and stop codons even though they have no neighbors
    STOP_CODONS = ['TAA', 'TAG', 'TGA']  # 3 stop codons
    all_codons = sorted(codon_neighbors.keys())
    
    # Add gap codon if not present (it won't be in codon_neighbors since it can't mutate)
    if '---' not in all_codons and '---' in codon_to_amino:
        all_codons = ['---'] + all_codons  # Add gap at the beginning
    
    # Add stop codons at the end (they can be proposed but immediately rejected)
    # Total: 1 gap + 61 coding + 3 stop = 65 codons
    for stop_codon in sorted(STOP_CODONS):
        if stop_codon not in all_codons:
            all_codons.append(stop_codon)
    
    codon_to_idx = {codon: idx for idx, codon in enumerate(all_codons)}
    num_codons = len(all_codons)
    num_aa = 21  # Including gap
    
    if device is None:
        device = torch.device('cpu')
    
    # Find maximum number of neighbor options for any (codon, pos, aa) combination
    max_neighbors = 1  # At least 1 for gap (itself)
    for codon in all_codons:
        # Skip gap codon - it has no neighbors, only itself
        if codon not in codon_neighbors:
            continue
        for pos in range(3):
            neighbor_codons = codon_neighbors[codon].get(pos, []) # e.g., for 'ATG' at pos 0: ['CTG', 'GTG', 'TTG']
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
        # Special handling for gap codon - it can only map to itself (no mutation)
        if codon not in codon_neighbors:
            if codon in codon_to_amino:
                gap_aa = codon_to_amino[codon]
                # For all positions, gap can only stay as gap
                for pos in range(3):
                    num_options[codon_idx, pos, gap_aa] = 1
                    mutation_lookup[codon_idx, pos, gap_aa, 0] = codon_idx
            continue
        
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


# E. coli codon usage frequencies
ECOLI_CODON_USAGE = {
    "TTT": 0.58, "TTC": 0.42,
    "TTA": 0.14, "TTG": 0.13,
    "TCT": 0.17, "TCC": 0.15, "TCA": 0.14, "TCG": 0.14,
    "TAT": 0.59, "TAC": 0.41,
    "TGT": 0.46, "TGC": 0.54, "TGG": 1.00,
    "CTT": 0.12, "CTC": 0.10, "CTA": 0.04, "CTG": 0.47,
    "CCT": 0.18, "CCC": 0.13, "CCA": 0.20, "CCG": 0.49,
    "CAT": 0.57, "CAC": 0.43,
    "CAA": 0.34, "CAG": 0.66,
    "CGT": 0.36, "CGC": 0.36, "CGA": 0.07, "CGG": 0.10,
    "ATT": 0.50, "ATC": 0.39, "ATA": 0.11,
    "ACT": 0.19, "ACC": 0.40, "ACA": 0.16, "ACG": 0.25,
    "AAT": 0.49, "AAC": 0.51,
    "AAA": 0.74, "AAG": 0.26,
    "AGT": 0.16, "AGC": 0.24, "AGA": 0.07, "AGG": 0.04,
    "GTT": 0.28, "GTC": 0.20, "GTA": 0.17, "GTG": 0.35,
    "GCT": 0.18, "GCC": 0.26, "GCA": 0.23, "GCG": 0.33,
    "GAT": 0.63, "GAC": 0.37,
    "GAA": 0.68, "GAG": 0.32, "ATG": 1.00,
    "GGT": 0.35, "GGC": 0.37, "GGA": 0.13, "GGG": 0.15,
    "---": 1.0
}

# Standard uniform codon usage based on degeneracy (1/number_of_codons_for_same_amino_acid)
STANDARD_CODON_USAGE = {
    # Phenylalanine (F) - 2 codons
    "TTT": 1.0/2, "TTC": 1.0/2,
    # Leucine (L) - 6 codons
    "TTA": 1.0/6, "TTG": 1.0/6, "CTT": 1.0/6, "CTC": 1.0/6, "CTA": 1.0/6, "CTG": 1.0/6,
    # Serine (S) - 6 codons
    "TCT": 1.0/6, "TCC": 1.0/6, "TCA": 1.0/6, "TCG": 1.0/6, "AGT": 1.0/6, "AGC": 1.0/6,
    # Tyrosine (Y) - 2 codons
    "TAT": 1.0/2, "TAC": 1.0/2,
    # Cysteine (C) - 2 codons
    "TGT": 1.0/2, "TGC": 1.0/2,
    # Tryptophan (W) - 1 codon
    "TGG": 1.0,
    # Proline (P) - 4 codons
    "CCT": 1.0/4, "CCC": 1.0/4, "CCA": 1.0/4, "CCG": 1.0/4,
    # Histidine (H) - 2 codons
    "CAT": 1.0/2, "CAC": 1.0/2,
    # Glutamine (Q) - 2 codons
    "CAA": 1.0/2, "CAG": 1.0/2,
    # Arginine (R) - 6 codons
    "CGT": 1.0/6, "CGC": 1.0/6, "CGA": 1.0/6, "CGG": 1.0/6, "AGA": 1.0/6, "AGG": 1.0/6,
    # Isoleucine (I) - 3 codons
    "ATT": 1.0/3, "ATC": 1.0/3, "ATA": 1.0/3,
    # Methionine (M) - 1 codon
    "ATG": 1.0,
    # Threonine (T) - 4 codons
    "ACT": 1.0/4, "ACC": 1.0/4, "ACA": 1.0/4, "ACG": 1.0/4,
    # Asparagine (N) - 2 codons
    "AAT": 1.0/2, "AAC": 1.0/2,
    # Lysine (K) - 2 codons
    "AAA": 1.0/2, "AAG": 1.0/2,
    # Valine (V) - 4 codons
    "GTT": 1.0/4, "GTC": 1.0/4, "GTA": 1.0/4, "GTG": 1.0/4,
    # Alanine (A) - 4 codons
    "GCT": 1.0/4, "GCC": 1.0/4, "GCA": 1.0/4, "GCG": 1.0/4,
    # Aspartic acid (D) - 2 codons
    "GAT": 1.0/2, "GAC": 1.0/2,
    # Glutamic acid (E) - 2 codons
    "GAA": 1.0/2, "GAG": 1.0/2,
    # Glycine (G) - 4 codons
    "GGT": 1.0/4, "GGC": 1.0/4, "GGA": 1.0/4, "GGG": 1.0/4,
    # Gap - 1 codon
    "---": 1.0
}


def build_codon_usage_tensor(
    codon_usage_dict: Dict[str, float],
    codon_to_idx: Dict[str, int],
    all_codons: List[str],
    device: torch.device = None
) -> torch.Tensor:
    """
    Build a 1D tensor of codon usage frequencies from a dictionary.
    
    Creates a tensor where codon_usage[codon_idx] contains the usage frequency
    for that codon according to the provided dictionary.
    
    Args:
        codon_usage_dict: Dictionary mapping codon strings to usage frequencies (e.g., ECOLI_CODON_USAGE)
        codon_to_idx: Dictionary mapping codon strings to indices
        all_codons: List of all codon strings
        device: Torch device for tensor allocation
    
    Returns:
        torch.Tensor: 1D tensor of shape (num_codons,) with usage frequencies
    
    Example:
        >>> ecoli_usage = build_codon_usage_tensor(ECOLI_CODON_USAGE, codon_to_idx, all_codons)
        >>> custom_usage = build_codon_usage_tensor({"ATG": 1.0, "TTT": 0.5, ...}, codon_to_idx, all_codons)
    """
    if device is None:
        device = torch.device('cpu')
    
    num_codons = len(all_codons)
    codon_usage_tensor = torch.zeros(num_codons, dtype=torch.float32, device=device)
    
    # Stop codons should have 0 usage (will be rejected anyway)
    STOP_CODONS = {'TAA', 'TAG', 'TGA'}
    
    # Fill tensor with usage frequencies
    for codon, idx in codon_to_idx.items():
        if codon in STOP_CODONS:
            codon_usage_tensor[idx] = 0.0  # Stop codons have 0 usage
        elif codon in codon_usage_dict:
            codon_usage_tensor[idx] = codon_usage_dict[codon]
        elif codon == '---':
            codon_usage_tensor[idx] = 1.0  # Gap has neutral usage
        else:
            raise ValueError(f"Codon '{codon}' not found in codon_usage_dict. All non-stop codons must have usage frequencies defined.")
    
    return codon_usage_tensor


def precompute_sampling_tensors(
    tokens: str,
    device: torch.device,
    codon_usage_dict: Dict[str, float] = None
) -> Dict:
    """
    Pre-compute all necessary tensors for GPU-optimized Gibbs sampling.
    
    This function builds all required data structures and tensors in one call,
    including codon mappings, neighbor networks, and lookup tables.
    
    Args:
        tokens: Amino acid alphabet string (e.g., from get_tokens("protein"))
        device: Torch device for tensor allocation (CPU/GPU)
        codon_usage_dict: Optional dictionary of codon usage frequencies. 
                         If None, uses STANDARD_CODON_USAGE (uniform based on degeneracy).
                         Can also use ECOLI_CODON_USAGE or any custom dictionary.
    
    Returns:
        Dict containing:
            - "codon_to_amino": Dict mapping codons to amino acid indices
            - "amino_to_codons": Dict mapping amino acid indices to codon lists
            - "codon_neighbors": Dict of codon nearest neighbors by position
            - "neighbor_counts": Dict of neighbor counts
            - "codon_neighbor_tensor": Tensor (num_codons, 3, q) for amino acid accessibility
            - "codon_neighbor_codon_tensor": Tensor (num_codons, 3, num_codons) for codon accessibility
            - "mutation_lookup": Tensor (num_codons, 3, q, max_neighbors) for mutations
            - "num_options": Tensor (num_codons, 3, q) for option counts
            - "codon_to_idx": Dict mapping codon strings to indices
            - "all_codons": List of all codons
            - "codon_usage": Tensor (num_codons,) with codon usage frequencies
    """
    num_amino_acids = len(tokens)
    
    # Build codon mappings
    codon_to_amino = build_codon_to_index_map(tokens) # e.g., {'ATA':9, 'ATC':9, 'ATT':9, 'ATG':12, ...}
    amino_to_codons = build_amino_to_codons_map(codon_to_amino) # e.g., 9: ['ATA', 'ATC', 'ATT'], 12: ['ATG'], ...
    
    # Build codon neighbor network
    # codon_neighbors: {codon: {position: [neighbor_codons]}} # e.g., 'ATG': {0: ['CTG', 'GTG', 'TTG'], 1: [...], 2: [...]}
    # neighbor_counts: {(codon, position): count} # e.g., ('ATG', 0): 3, ('ATG', 1): 3, ('ATG', 2): 3
    codon_neighbors, neighbor_counts = build_codon_neighbors()

    # Build GPU tensors for fast sampling
    codon_neighbor_tensor, codon_neighbor_codon_tensor, codon_to_idx, all_codons = build_codon_neighbor_tensor(
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
    
    # Build codon usage tensor (default to standard uniform if not provided)
    if codon_usage_dict is None:
        codon_usage_dict = STANDARD_CODON_USAGE
    
    codon_usage = build_codon_usage_tensor(
        codon_usage_dict,
        codon_to_idx,
        all_codons,
        device
    )
    
    # Build codon_to_aa_idx tensor: maps each codon index to its amino acid index
    # This eliminates the slow CPU loop in gibbs_step_batch
    # Stop codons don't have amino acids, map them to gap (index 0) - they'll be rejected anyway
    STOP_CODONS = {'TAA', 'TAG', 'TGA'}
    num_codons = len(all_codons)
    codon_to_aa_idx = torch.zeros(num_codons, dtype=torch.long, device=device)
    for codon_idx, codon_str in enumerate(all_codons):
        if codon_str in STOP_CODONS:
            codon_to_aa_idx[codon_idx] = 0  # Map stop codons to gap (will be rejected)
        else:
            codon_to_aa_idx[codon_idx] = codon_to_amino[codon_str]
    
    # Pre-compute tensors for Metropolis sampling (bottleneck optimization)
    # Find gap codon index (gap is '---')
    gap_codon_str = '---'
    gap_codon_idx = all_codons.index(gap_codon_str) if gap_codon_str in all_codons else 0
    
    # Build non-gap codon indices tensor (ALL 64 codons except gap, INCLUDING stop codons)
    # Stop codons can be proposed but will be rejected in acceptance calculation
    non_gap_codon_indices = [i for i, codon in enumerate(all_codons) if codon != gap_codon_str]
    non_gap_codon_tensor = torch.tensor(non_gap_codon_indices, dtype=torch.long, device=device)
    
    # Build stop codon indices tensor for rejection
    STOP_CODONS = {'TAA', 'TAG', 'TGA'}
    stop_codon_indices = [i for i, codon in enumerate(all_codons) if codon in STOP_CODONS]
    stop_codon_indices_tensor = torch.tensor(stop_codon_indices, dtype=torch.long, device=device)
    
    # GPU OPTIMIZATION: Pre-compute boolean mask for stop codons (faster than torch.isin)
    stop_codon_mask = torch.zeros(num_codons, dtype=torch.bool, device=device)
    stop_codon_mask[stop_codon_indices] = True
    
    # GPU OPTIMIZATION: Pre-compute one-hot encoding for all codons (avoids repeated one_hot() calls)
    # Shape: (num_codons, num_amino_acids) - direct lookup codon_to_aa_onehot[codon_idx]
    num_aa = len(tokens)
    codon_to_aa_onehot = torch.zeros(num_codons, num_aa, dtype=torch.float32, device=device)
    for codon_idx in range(num_codons):
        aa_idx = codon_to_aa_idx[codon_idx].item()
        codon_to_aa_onehot[codon_idx, aa_idx] = 1.0
    
    # GPU OPTIMIZATION: Pre-allocate tensor for gap proposals (avoids torch.full() every iteration)
    gap_tensor = torch.full((max(1000, num_codons),), gap_codon_idx, dtype=torch.long, device=device)
    
    # GPU OPTIMIZATION: Pre-compute log(codon_usage) to avoid repeated log() calls
    log_codon_usage = torch.log(codon_usage + 1e-10)
    
    # Pre-compute tensors for translate_to_dna_uniform (bottleneck optimization #1)
    # Build amino_to_codons_tensor: for each amino acid, store list of codon indices
    # Structure: (num_aa, max_codons_per_aa) with -1 padding for unused slots
    max_codons_per_aa = max(len(codons) for codons in amino_to_codons.values())
    amino_to_codons_tensor = torch.full(
        (num_amino_acids, max_codons_per_aa), 
        -1, 
        dtype=torch.long, 
        device=device
    )
    amino_to_num_codons = torch.zeros(num_amino_acids, dtype=torch.long, device=device)
    
    for aa_idx, codon_list in amino_to_codons.items():
        num_codons_for_aa = len(codon_list)
        amino_to_num_codons[aa_idx] = num_codons_for_aa
        for i, codon_str in enumerate(codon_list):
            amino_to_codons_tensor[aa_idx, i] = codon_to_idx[codon_str]
    
    return {
        "codon_to_amino": codon_to_amino,
        "amino_to_codons": amino_to_codons,
        "codon_neighbors": codon_neighbors,
        "neighbor_counts": neighbor_counts,
        "codon_neighbor_tensor": codon_neighbor_tensor,
        "codon_neighbor_codon_tensor": codon_neighbor_codon_tensor,
        "mutation_lookup": mutation_lookup,
        "num_options": num_options,
        "codon_to_idx": codon_to_idx,
        "all_codons": all_codons,
        "codon_usage": codon_usage,
        "codon_to_aa_idx": codon_to_aa_idx,
        "gap_codon_idx": gap_codon_idx,
        "non_gap_codon_tensor": non_gap_codon_tensor,
        "stop_codon_indices": stop_codon_indices_tensor,
        "amino_to_codons_tensor": amino_to_codons_tensor,
        "amino_to_num_codons": amino_to_num_codons,
        # GPU optimizations (pre-computed tensors)
        "stop_codon_mask": stop_codon_mask,
        "codon_to_aa_onehot": codon_to_aa_onehot,
        "gap_tensor": gap_tensor,
        "log_codon_usage": log_codon_usage
    }
