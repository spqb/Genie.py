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
- **GPU-Accelerated**: Full CUDA support with PyTorch JIT compilation (2-3x speedup)
- **Codon-Aware Sampling**: Biologically realistic single-nucleotide mutations at DNA level
- **Hybrid MCMC**: Combined Metropolis-Hastings and Gibbs sampling for better mixing
- **Reference-Based**: Optional convergence tracking against real sequence data
- **Flexible Input**: Start from existing sequences or random initialization

### Technical Highlights
- Fully vectorized GPU kernels with zero CPU loops
- Pre-computed codon mutation networks for O(1) neighbor lookups
- Batched random number generation for improved GPU efficiency
- Masked operations eliminate branching overhead
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
pip install -e .
```

This installs two command-line tools:
- `genie` - Codon-aware evolution
- `genie-aa` - Amino acid evolution

---

## Quick Start

### 1. Sample from a DCA Model (Codon-Aware)

```bash
genie \
  -p params.dat \
  -n 1000 \
  --num_iterations 50000 \
  -o output_folder
```

### 2. Sample with Reference Data Tracking

```bash
genie \
  -d reference_data.fasta \
  -p params.dat \
  -n 1000 \
  --num_iterations 100000 \
  -o output_folder
```

### 3. Amino Acid-Only Sampling

```bash
genie-aa \
  -p params.dat \
  -n 1000 \
  --num_iterations 50000 \
  -o output_folder
```

---

## Usage

### Genie (Codon Evolution)

**Basic Sampling:**
```bash
genie -p <params.dat> -n <num_sequences> --num_iterations <iterations> -o <output_dir>
```

**With Initial Sequences:**
```bash
genie -c <init_sequences.fasta> -p <params.dat> -n 500 --num_iterations 100000 -o results/
```

**With Reference Data:**
```bash
genie -d <reference.fasta> -p <params.dat> -n 1000 --num_iterations 100000 -o results/
```

**Replicate Single Sequence:**
```bash
genie -c <sequences.fasta> --seq_index 42 -n 1000 --num_iterations 50000 -o results/
```

### Genie-AA (Amino Acid Evolution)

**Basic Sampling:**
```bash
genie-aa -p <params.dat> -n <num_sequences> --num_iterations <iterations> -o <output_dir>
```

**With Reference Tracking:**
```bash
genie-aa -d <reference.fasta> -p <params.dat> -n 1000 --num_iterations 100000 -o results/
```

---

## Command-Line Arguments

### Common Arguments (Both Tools)

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--path_params` | `-p` | Yes | - | DCA model parameters file (.dat) |
| `--num_chains` | `-n` | Yes* | - | Number of sequences to generate |
| `--num_iterations` | | Yes | - | Number of MCMC iterations |
| `--output` | `-o` | Yes | - | Output directory path |
| `--path_chains` | `-c` | No | None | Initial sequences (FASTA) |
| `--data` | `-d` | No | None | Reference data for convergence tracking |
| `--seq_index` | | No | None | Index of sequence to replicate (with `-c`) |
| `--device` | | No | auto | Device: 'cuda' or 'cpu' |
| `--dtype` | | No | float32 | Data type: float32 or float64 |
| `--alphabet` | | No | protein | Alphabet type (genie-aa only) |

### Genie-Specific Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--p_metropolis` | | 0.5 | Metropolis vs Gibbs ratio (0-1) |

**Note:** When using `--path_chains`, `--num_chains` is optional (uses all loaded sequences if not specified)

---

## Output Files

### Sampling Log (if reference data provided)

**File:** `sampling.log`

CSV file with convergence metrics:

| Column | Description |
|--------|-------------|
| `iteration` | MCMC iteration number |
| `pearson` | Pearson correlation with reference data |

**Example:**
```csv
iteration,pearson
1000,0.8234
2000,0.8567
3000,0.8823
...
```

### Console Output

Real-time progress with:
- Iterations per second
- Pearson correlation (if reference data)
- Gap frequency statistics (if reference data)
- Total execution time

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

### Generate 10,000 Sequences

```bash
genie \
  -p example_data/pf76/params.dat \
  -n 10000 \
  --num_iterations 100000 \
  -o results/pf76_10k
```

### Track Convergence Against Real Data

```bash
genie \
  -d example_data/pf76/PF00076_mgap6.fasta \
  -p example_data/pf76/params.dat \
  -n 5000 \
  --num_iterations 200000 \
  -o results/pf76_convergence
```

---

## Citation

If you use Genie 2.0 in your research, please cite:

```bibtex
@software{genie2024,
  title={Genie 2.0: GPU-Accelerated Codon-Aware Sequence Evolution},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Genie.py}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built on the [adabmDCA](https://github.com/spqb/adabmDCA) library
- PyTorch team for excellent GPU optimization tools