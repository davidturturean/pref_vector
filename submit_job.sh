#!/bin/bash
# Quick submission script for MIT cluster

# Check if project directory is specified
if [ -z "$1" ]; then
    echo "Usage: $0 <project_directory_on_cluster>"
    echo "Example: $0 /nfs/home2/davidct/pref_vector"
    exit 1
fi

PROJECT_DIR="$1"

# Update the run_cluster.sh with correct path
sed -i "s|cd /nfs/home2/davidct/pref_vector|cd $PROJECT_DIR|" run_cluster.sh

# Submit job
echo "Submitting job to cluster..."
echo "Project directory: $PROJECT_DIR"
sbatch run_cluster.sh

echo "Job submitted! Check status with: squeue -u $USER"
echo "Monitor logs with: tail -f logs/slurm_*.out"
