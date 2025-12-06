"""
Codon utilities for genetic sequence manipulation.
"""

from itertools import product


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
