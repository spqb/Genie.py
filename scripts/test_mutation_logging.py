"""
Test script for validating the checkpoint-based mutation logging system.

This script runs both Genie and Genie-AA with small test cases and validates
that the reconstruction from mutation logs produces identical final chains.
"""
import os
import sys
import subprocess
import tempfile
import shutil

def run_test(script_name, test_name, command, description):
    """Run a test and return success status."""
    print("\n" + "="*80)
    print(f"TEST: {test_name}")
    print("="*80)
    print(f"Description: {description}")
    print(f"Command: {' '.join(command)}")
    print("-"*80)
    
    try:
        # Run the command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"✗ FAILED: Command returned non-zero exit code {result.returncode}")
            print("STDERR:")
            print(result.stderr)
            return False
        
        print(f"✓ PASSED")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"✗ FAILED: Command timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"✗ FAILED: Exception occurred: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("GENIE MUTATION LOGGING VALIDATION TESTS")
    print("="*80)
    
    # Check if example data exists
    example_chains = "example_data/chains.fasta"
    if not os.path.exists(example_chains):
        print(f"ERROR: Example data not found: {example_chains}")
        print("Please ensure example_data/chains.fasta exists")
        sys.exit(1)
    
    # Check if DCA params exist (you'll need to update this path)
    # For now, we'll skip if not available
    test_params = "example_data/params.dat"  # Update this path as needed
    
    results = []
    
    # ========================================================================
    # Test 1: Genie-AA with small number of iterations
    # ========================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "genie_aa_test")
        
        # Run Genie-AA
        command = [
            "python", "-m", "Genie_aa.main",
            "--path_chains", example_chains,
            "--path_params", test_params,
            "--output", output_dir,
            "--num_chains", "5",
            "--num_iterations", "1000",
            "--save_steps", "100"
        ]
        
        success = run_test(
            "Genie-AA",
            "Genie-AA Small Test (1000 iterations, 5 chains)",
            command,
            "Run Genie-AA with checkpoint-based mutation logging"
        )
        
        if success and os.path.exists(output_dir):
            # Verify files were created
            initial_file = os.path.join(output_dir, "initial_chains.fasta")
            mutation_log = os.path.join(output_dir, "mutation_log.csv")
            final_file = os.path.join(output_dir, "final_chains.fasta")
            
            if not os.path.exists(initial_file):
                print(f"✗ FAILED: Initial chains file not created")
                success = False
            elif not os.path.exists(mutation_log):
                print(f"✗ FAILED: Mutation log file not created")
                success = False
            elif not os.path.exists(final_file):
                print(f"✗ FAILED: Final chains file not created")
                success = False
            else:
                # Run reconstruction
                recon_command = [
                    "python", "scripts/reconstruct_chains.py",
                    output_dir,
                    "--alphabet", "protein"
                ]
                
                recon_success = run_test(
                    "Reconstruct",
                    "Genie-AA Reconstruction Validation",
                    recon_command,
                    "Reconstruct chains from mutation log and validate"
                )
                
                success = success and recon_success
        
        results.append(("Genie-AA Test", success))
    
    # ========================================================================
    # Test 2: Genie with small number of iterations
    # ========================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "genie_test")
        
        # Run Genie
        command = [
            "python", "-m", "Genie.main",
            "--path_chains", example_chains,
            "--path_params", test_params,
            "--output", output_dir,
            "--num_chains", "5",
            "--num_iterations", "1000",
            "--save_steps", "100",
            "--p_metropolis", "0.5"
        ]
        
        success = run_test(
            "Genie",
            "Genie Small Test (1000 iterations, 5 chains)",
            command,
            "Run Genie with checkpoint-based mutation logging"
        )
        
        if success and os.path.exists(output_dir):
            # Verify files were created
            initial_file = os.path.join(output_dir, "initial_chains.fasta")
            mutation_log = os.path.join(output_dir, "mutation_log.csv")
            final_file = os.path.join(output_dir, "final_chains.fasta")
            
            if not os.path.exists(initial_file):
                print(f"✗ FAILED: Initial chains file not created")
                success = False
            elif not os.path.exists(mutation_log):
                print(f"✗ FAILED: Mutation log file not created")
                success = False
            elif not os.path.exists(final_file):
                print(f"✗ FAILED: Final chains file not created")
                success = False
            else:
                # Run reconstruction
                recon_command = [
                    "python", "scripts/reconstruct_chains.py",
                    output_dir,
                    "--alphabet", "protein"
                ]
                
                recon_success = run_test(
                    "Reconstruct",
                    "Genie Reconstruction Validation",
                    recon_command,
                    "Reconstruct chains from mutation log and validate"
                )
                
                success = success and recon_success
        
        results.append(("Genie Test", success))
    
    # ========================================================================
    # Test 3: Edge case - save_steps equals num_iterations
    # ========================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "genie_aa_edge_test")
        
        command = [
            "python", "-m", "Genie_aa.main",
            "--path_chains", example_chains,
            "--path_params", test_params,
            "--output", output_dir,
            "--num_chains", "3",
            "--num_iterations", "100",
            "--save_steps", "100"
        ]
        
        success = run_test(
            "Genie-AA Edge Case",
            "Genie-AA Edge Case (save_steps == num_iterations)",
            command,
            "Test when checkpoint interval equals total iterations"
        )
        
        if success and os.path.exists(output_dir):
            recon_command = [
                "python", "scripts/reconstruct_chains.py",
                output_dir,
                "--alphabet", "protein"
            ]
            
            recon_success = run_test(
                "Reconstruct",
                "Genie-AA Edge Case Reconstruction",
                recon_command,
                "Validate edge case reconstruction"
            )
            
            success = success and recon_success
        
        results.append(("Genie-AA Edge Case", success))
    
    # ========================================================================
    # Print Summary
    # ========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<40} {status}")
        if not success:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
