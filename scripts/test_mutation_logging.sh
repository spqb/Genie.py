#!/bin/bash

# Simple manual test for mutation logging system
# This script helps you quickly test the new checkpoint-based mutation logging

echo "==========================================================================="
echo "Genie Mutation Logging Test"
echo "==========================================================================="
echo ""

# Check if example data exists
if [ ! -f "example_data/chains.fasta" ]; then
    echo "ERROR: example_data/chains.fasta not found"
    echo "Please provide example data first"
    exit 1
fi

# You need to provide a DCA parameters file
# Update this path to your actual DCA parameters file
PARAMS_FILE="${1:-example_data/params.dat}"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameters file not found: $PARAMS_FILE"
    echo ""
    echo "Usage: $0 <path_to_params_file>"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/your/params.dat"
    exit 1
fi

# Create temporary output directory
OUTPUT_DIR="test_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Parameters file: $PARAMS_FILE"
echo ""

# ========================================================================
# Test 1: Genie-AA
# ========================================================================
echo "==========================================================================="
echo "TEST 1: Genie-AA"
echo "==========================================================================="
echo "Running Genie-AA with 5 chains, 1000 iterations, checkpoint every 100..."
echo ""

python -m Genie_aa.main \
    --path_chains example_data/chains.fasta \
    --path_params "$PARAMS_FILE" \
    --output "${OUTPUT_DIR}/genie_aa" \
    --num_chains 5 \
    --num_iterations 1000 \
    --save_steps 100

if [ $? -ne 0 ]; then
    echo "✗ Genie-AA run failed"
    exit 1
fi

echo ""
echo "Verifying output files..."
if [ ! -f "${OUTPUT_DIR}/genie_aa/initial_chains.fasta" ]; then
    echo "✗ initial_chains.fasta not created"
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/genie_aa/mutation_log.csv" ]; then
    echo "✗ mutation_log.csv not created"
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/genie_aa/final_chains.fasta" ]; then
    echo "✗ final_chains.fasta not created"
    exit 1
fi
echo "✓ All output files created"

echo ""
echo "Reconstructing chains from mutation log..."
python scripts/reconstruct_chains.py "${OUTPUT_DIR}/genie_aa" --alphabet protein

if [ $? -eq 0 ]; then
    echo "✓ Genie-AA reconstruction validation PASSED"
else
    echo "✗ Genie-AA reconstruction validation FAILED"
    exit 1
fi

# ========================================================================
# Test 2: Genie
# ========================================================================
echo ""
echo "==========================================================================="
echo "TEST 2: Genie (codon-aware)"
echo "==========================================================================="
echo "Running Genie with 5 chains, 1000 iterations, checkpoint every 100..."
echo ""

python -m Genie.main \
    --path_chains example_data/chains.fasta \
    --path_params "$PARAMS_FILE" \
    --output "${OUTPUT_DIR}/genie" \
    --num_chains 5 \
    --num_iterations 1000 \
    --save_steps 100 \
    --p_metropolis 0.5

if [ $? -ne 0 ]; then
    echo "✗ Genie run failed"
    exit 1
fi

echo ""
echo "Verifying output files..."
if [ ! -f "${OUTPUT_DIR}/genie/initial_chains.fasta" ]; then
    echo "✗ initial_chains.fasta not created"
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/genie/mutation_log.csv" ]; then
    echo "✗ mutation_log.csv not created"
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/genie/final_chains.fasta" ]; then
    echo "✗ final_chains.fasta not created"
    exit 1
fi
echo "✓ All output files created"

echo ""
echo "Reconstructing chains from mutation log..."
python scripts/reconstruct_chains.py "${OUTPUT_DIR}/genie" --alphabet protein

if [ $? -eq 0 ]; then
    echo "✓ Genie reconstruction validation PASSED"
else
    echo "✗ Genie reconstruction validation FAILED"
    exit 1
fi

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "==========================================================================="
echo "ALL TESTS PASSED!"
echo "==========================================================================="
echo ""
echo "Test output saved in: $OUTPUT_DIR"
echo ""
echo "You can inspect the mutation logs:"
echo "  - Genie-AA: ${OUTPUT_DIR}/genie_aa/mutation_log.csv"
echo "  - Genie:    ${OUTPUT_DIR}/genie/mutation_log.csv"
echo ""
