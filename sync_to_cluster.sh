#!/bin/bash

# Streamlined sync to MIT Engaging cluster
# Usage: ./sync_to_cluster.sh

set -e  # Exit on any error

# Configuration
CLUSTER_USER="davidct"
CLUSTER_HOST="eofe7.mit.edu"
REMOTE_PATH="/home/davidct/pref_vector"
REMOTE_TARGET="$CLUSTER_USER@$CLUSTER_HOST"

echo "🚀 Syncing pref_vector to MIT cluster..."
echo "   Target: $REMOTE_TARGET:$REMOTE_PATH"
echo ""

# Check if we're in the right directory
if [ ! -f "run_experiment.py" ]; then
    echo "❌ Error: run_experiment.py not found"
    echo "   Make sure you're in the pref_vector directory"
    exit 1
fi

# Sync minimal version (exclude large/unnecessary files)
echo "📦 Syncing files..."
rsync -av --progress \
    --exclude='pref_vector_env/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='*.log' \
    --exclude='tmp/' \
    --exclude='temp/' \
    --exclude='.DS_Store' \
    --exclude='*.tmp' \
    --exclude='results/pref_vector_exp_*/' \
    --exclude='data/*/data-*.arrow' \
    --exclude='*.tar.gz' \
    --delete-excluded \
    --filter='protect logs/' \
    --filter='protect results/' \
    --filter='protect data/' \
    . $REMOTE_TARGET:$REMOTE_PATH/

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Sync completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. SSH to cluster: ssh $REMOTE_TARGET"
    echo "   2. Navigate: cd $REMOTE_PATH"
    echo "   3. Submit job: sbatch run_cluster.sh"
    echo "   4. Monitor: squeue -u $CLUSTER_USER"
    echo "   5. View logs: tail -f logs/slurm_*.out"
    echo ""
    echo "💡 Quick commands:"
    echo "   ssh $REMOTE_TARGET 'cd $REMOTE_PATH && sbatch run_cluster.sh'"
    echo "   ssh $REMOTE_TARGET 'squeue -u $CLUSTER_USER'"
else
    echo ""
    echo "❌ Sync failed!"
    echo "   Check your network connection and cluster access"
    exit 1
fi