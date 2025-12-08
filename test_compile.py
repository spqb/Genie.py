#!/usr/bin/env python
"""
Quick test of torch.compile() with evolve_sequences.
Verifies compilation works and measures speedup.
"""
import torch
import time
import sys
sys.path.insert(0, '/home/robertonetti/Desktop/Github/Genie.py')

from Genie.core.evolution import evolve_sequences
from Genie.utils.codon_utils import precompute_sampling_tensors

print("Testing torch.compile() with evolve_sequences...")
print(f"PyTorch version: {torch.__version__}")
print()

# Setup small test case
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Small test dimensions
N = 100  # chains
L = 50   # sequence length
q = 21   # amino acids

# Create dummy data
chains = torch.randn(N, L, q, device=device).softmax(dim=-1)
dna_chains = torch.randint(0, 61, (N, L), device=device)

# Create dummy params
params = {
    "bias": torch.randn(L, q, device=device),
    "coupling_matrix": torch.randn(L, q, L, q, device=device) * 0.1,
    "gap_codon_idx": 61,  # Python int, not tensor!
    "non_gap_codon_tensor": torch.arange(61, device=device),
    "codon_to_aa_idx": torch.randint(0, 21, (62,), device=device),
}

# Create dummy sampling tensors (simplified)
num_codons = 62
codon_neighbor_tensor = torch.randint(0, 2, (num_codons, 3, q), device=device).float()
codon_neighbor_codon_tensor = torch.randint(0, 2, (num_codons, 3, num_codons), device=device).bool()
mutation_lookup = torch.randint(0, num_codons, (num_codons, 3, q, 10), device=device)
num_options = torch.randint(1, 10, (num_codons, 3, q), device=device)
codon_usage = torch.rand(num_codons, device=device)
codon_usage = codon_usage / codon_usage.sum()

# Test uncompiled version
print("\n=== Uncompiled version ===")
t0 = time.time()
for i in range(10):
    chains_copy = chains.clone()
    dna_copy = dna_chains.clone()
    evolve_sequences(
        chains=chains_copy,
        dna_chains=dna_copy,
        params=params,
        codon_neighbor_tensor=codon_neighbor_tensor,
        codon_neighbor_codon_tensor=codon_neighbor_codon_tensor,
        mutation_lookup=mutation_lookup,
        num_options=num_options,
        codon_usage=codon_usage,
        device=device,
        p=0.5
    )
t_uncompiled = (time.time() - t0) / 10
print(f"Average time per call: {t_uncompiled*1000:.2f}ms")

# Test compiled version
print("\n=== Compiled version (max-autotune) ===")
try:
    evolve_compiled = torch.compile(
        evolve_sequences,
        mode='max-autotune',
        fullgraph=True,
        dynamic=False
    )
    
    print("Warmup (compilation)...")
    t_warmup = time.time()
    chains_copy = chains.clone()
    dna_copy = dna_chains.clone()
    evolve_compiled(
        chains=chains_copy,
        dna_chains=dna_copy,
        params=params,
        codon_neighbor_tensor=codon_neighbor_tensor,
        codon_neighbor_codon_tensor=codon_neighbor_codon_tensor,
        mutation_lookup=mutation_lookup,
        num_options=num_options,
        codon_usage=codon_usage,
        device=device,
        p=0.5
    )
    warmup_time = time.time() - t_warmup
    print(f"Compilation time: {warmup_time:.2f}s")
    
    print("\nRunning compiled benchmark...")
    t0 = time.time()
    for i in range(10):
        chains_copy = chains.clone()
        dna_copy = dna_chains.clone()
        evolve_compiled(
            chains=chains_copy,
            dna_chains=dna_copy,
            params=params,
            codon_neighbor_tensor=codon_neighbor_tensor,
            codon_neighbor_codon_tensor=codon_neighbor_codon_tensor,
            mutation_lookup=mutation_lookup,
            num_options=num_options,
            codon_usage=codon_usage,
            device=device,
            p=0.5
        )
    t_compiled = (time.time() - t0) / 10
    print(f"Average time per call: {t_compiled*1000:.2f}ms")
    
    speedup = t_uncompiled / t_compiled
    print(f"\n✅ Speedup: {speedup:.2f}x")
    
    if speedup > 1.5:
        print("🎉 Compilation working well!")
    else:
        print("⚠️  Speedup lower than expected (might need larger problem)")
    
except Exception as e:
    print(f"❌ Compilation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete!")
