# Auto-activate conda environment for this project
if [ -n "$CONDA_EXE" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate genie_env 2>/dev/null
fi
