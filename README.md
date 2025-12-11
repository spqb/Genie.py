# Genie 2.0

**GPU-Accelerated MCMC Sampling for Protein Sequences with Codon-Level Mutations**

Genie 2.0 is a high-performance tool for generating protein sequences using Direct Coupling Analysis (DCA) models combined with biologically realistic codon substitution dynamics. It implements efficient MCMC sampling on GPUs with two variants:

- **Genie**: DNA codon-aware evolution with Metropolis-Gibbs sampling
- **Genie-AA**: Amino acid-only evolution with standard Gibbs sampling

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Output Files](#output-files)
- [Algorithm Overview](#algorithm-overview)
- [Performance](#performance)
- [Requirements](#requirements)
- [Examples](#examples)
- [Citation](#citation)
- [License](#license)

---

## Features

### Core Capabilities
- **Codon-Aware Sampling**: Biologically realistic single-nucleotide mutations at DNA level
- **Hybrid MCMC**: Combined Metropolis-Hastings and Gibbs sampling for better mixing
- **Reference-Based**: Optional convergence tracking against real sequence data
- **GPU-Accelerated**: Full CUDA support with PyTorch JIT compilation (2-3x speedup)
- **Flexible Input**: Start from existing sequences or random initialization

### Technical Highlights
- Fully vectorized GPU kernels with zero CPU loops
- Pre-computed codon mutation networks for O(1) neighbor lookups
- Batched random number generation for improved GPU efficiency
- Real-time Pearson correlation tracking for convergence monitoring

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended) or CPU
- adabmDCA library

### Install from Source

```bash
git clone https://github.com/yourusername/Genie.py.git
cd Genie.py
pip install .
```

This installs two command-line tools:
- `genie` - Codon-aware evolution
- `genie-aa` - Amino acid evolution

---

## Quick Start

### Codon-Aware Evolution

```bash
genie -p params.dat -n 1000 --num_iterations 50000 -o output_folder
```

### Amino Acid Evolution

```bash
genie-aa -p params.dat -n 1000 --num_iterations 50000 -o output_folder
```

---

## Usage

### Genie (Codon-Aware Evolution)

```bash
# Generate sequences from scratch
genie -p params.dat -n 1000 --num_iterations 50000 -o results/

# Start from existing sequences
genie -c init_sequences.fasta -p params.dat --num_iterations 50000 -o results/
```

### Genie-AA (Amino Acid Evolution)

```bash
# Generate sequences from scratch
genie-aa -p params.dat -n 1000 --num_iterations 50000 -o results/

# Start from existing sequences
genie-aa -c init_sequences.fasta -p params.dat --num_iterations 50000 -o results/
```

### Reconstruction Tools

```bash
# Reconstruct final sequences from mutation log
reconstruct_chains results/

# Reconstruct sequences at specific timesteps
reconstruct_at_timesteps results/ --timesteps "0,100,500,1000"
```

### Python API

```python
from Genie import reconstruct_at_timesteps, reconstruct_chains_from_log
from adabmDCA.fasta import get_tokens

# Reconstruct sequences at specific timesteps
sequences = reconstruct_at_timesteps(
    initial_chains_file="results/initial_chains.fasta",
    mutation_log_file="results/mutation_log.csv",
    timesteps=[0, 100, 500, 1000],
    alphabet="protein"
)
# Returns: torch.Tensor of shape (len(timesteps), n_chains, L)

# Reconstruct and validate final sequences
tokens = get_tokens(alphabet="protein")
reconstructed_seqs, headers = reconstruct_chains_from_log(
    initial_chains_file="results/initial_chains.fasta",
    mutation_log_file="results/mutation_log.csv",
    tokens=tokens
)
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--path_params` | `-p` | DCA model parameters file (.dat) |
| `--num_iterations` | | Number of MCMC iterations |

### Optional Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--output` | `-o` | `DCA_evolution` | Output directory |
| `--num_chains` | `-n` | None | Number of sequences (required if not using `-c`) |
| `--path_chains` | `-c` | None | Initial sequences (FASTA format) |
| `--seq_index` | | None | Replicate single sequence from `-c` file |
| `--save_steps` | | `100` | Checkpoint interval or comma-separated list (e.g., "100,500,1000") |
| `--device` | | auto | Device: 'cuda' or 'cpu' |
| `--dtype` | | float32 | Data type: float32 or float64 |

### Genie-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--p_metropolis` | 0.5 | Metropolis vs Gibbs ratio (0.0-1.0) |

### Genie-AA Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--alphabet` | protein | Alphabet type: 'protein', 'rna', 'dna', or custom |

### Reconstruction Tool Arguments

**reconstruct_chains**: Takes output folder as positional argument

**reconstruct_at_timesteps**: 
- `folder` - Output folder (positional)
- `--timesteps` - Comma-separated list (e.g., "0,100,500,1000")

---

## Output Files

All files are saved in the output directory specified by `-o`.

### Generated Files

| File | Description |
|------|-------------|
| `initial_chains.fasta` | Starting sequences (before evolution) |
| `final_chains.fasta` | Final sequences (after all iterations) |
| `mutation_log.csv` | Log of all mutations at checkpoints |

### Mutation Log Format

**File:** `mutation_log.csv`

CSV file tracking mutations at checkpoints:

| Column | Description |
|--------|-------------|
| `iteration` | Checkpoint iteration number |
| `chain_id` | Sequence identifier |
| `position` | Position in sequence (0-indexed) |
| `new_aa` | New amino acid at this position |

**Example:**
```csv
iteration,chain_id,position,new_aa
100,seq_0,15,A
100,seq_0,42,G
100,seq_1,23,L
200,seq_0,15,V
...
```

### Console Output

Real-time progress showing:
- Iteration number and speed (iter/sec)
- Elapsed time
- Compilation status (first iteration)

---

## Algorithm Overview

### Genie (Codon Evolution)

1. **Initialization**: Load DCA model, build codon mutation network
2. **Sequence Translation**: Convert amino acids to codons
3. **MCMC Sampling**: Hybrid Metropolis-Gibbs with codon mutations
4. **Convergence Tracking**: Optional Pearson correlation monitoring

### Genie-AA (Amino Acid Only)

1. **Initialization**: Load DCA model
2. **Gibbs Sampling**: Standard position-wise sampling
3. **Convergence Tracking**: Optional correlation monitoring

---

## Performance

**Hardware:** NVIDIA RTX 4090, 1000 sequences, L=100

| Mode | Iterations/sec | Speedup |
|------|----------------|---------|
| Genie (compiled) | ~45-50 | 2.5x |
| Genie (eager) | ~18-20 | 1.0x |
| Genie-AA (compiled) | ~120-140 | 6.5x |

**Note:** First iteration includes ~10-30s JIT compilation overhead

---

## Requirements

```txt
torch>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
adabmDCA>=1.0.0
```

**Hardware:**
- Minimum: CPU with 4GB RAM
- Recommended: NVIDIA GPU (8GB+ VRAM) with CUDA 11.7+

---

## Examples

### Basic Evolution

```bash
# Generate 1000 sequences with codon awareness
genie -p example_data/pf76/params.dat -n 1000 --num_iterations 50000 -o results/pf76

# Generate amino acid sequences only
genie-aa -p example_data/pf76/params.dat -n 1000 --num_iterations 50000 -o results/pf76_aa
```

### Custom Checkpoints

```bash
# Save mutations at specific iterations
genie -p params.dat -n 1000 --num_iterations 10000 --save_steps "100,500,1000,5000,10000" -o results/

# Reconstruct sequences at those timesteps
reconstruct_at_timesteps results/ --timesteps "0,100,500,1000,5000,10000"
```

---

## Citation

This software is based on the following article:

```bibtex
@article{
doi:10.1073/pnas.2406807121,
author = {Leonardo Di Bari  and Matteo Bisardi  and Sabrina Cotogno  and Martin Weigt  and Francesco Zamponi },
title = {Emergent time scales of epistasis in protein evolution},
journal = {Proceedings of the National Academy of Sciences},
volume = {121},
number = {40},
pages = {e2406807121},
year = {2024},
doi = {10.1073/pnas.2406807121},
URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2406807121},
eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2406807121},
}
```

A Julia version of Genie is also available: [Genie.jl](https://github.com/spqb/Genie.jl)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built on the [adabmDCA](https://github.com/spqb/adabmDCA) library
- PyTorch team for excellent GPU optimization tools