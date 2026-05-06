#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 40:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A YOUR_ALLOCATION
set -x

WORK_DIR="${PROJECT_ROOT:-/path/to/dpr-bag}/dpr-bag/"
OLLAMA_DIR="${OLLAMA_DIR:-/path/to/ollama_setup}"
CONDA_PROFILE="${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-dpr_bag}"

export OLLAMA_LIBRARY_PATH="$OLLAMA_DIR/lib/ollama"     


cd $WORK_DIR
mkdir -p logs 

source $CONDA_PROFILE
conda activate $CONDA_ENV

module load cuda/12.4.0

if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    echo "Auto-detected CUDA_HOME: $CUDA_HOME"
fi


export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Debug: CUDA_HOME is $CUDA_HOME"
echo "Debug: LD_LIBRARY_PATH is $LD_LIBRARY_PATH"

nvidia-smi

export OLLAMA_MODELS="$OLLAMA_DIR/ollama_models"
export HOME=$OLLAMA_DIR

export OLLAMA_TMPDIR="$WORK_DIR/ollama_tmp"
mkdir -p $OLLAMA_TMPDIR

export OLLAMA_DEBUG=1


export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_CONTEXT=4096 
export OLLAMA_KEEP_ALIVE=-1

echo "Starting Ollama Server..."
$OLLAMA_DIR/bin/ollama serve > logs/ollama_server.log 2>&1 &
OLLAMA_PID=$! 

echo "Waiting for Ollama to be ready..."
for i in {1..20}; do
    if curl -s http://127.0.0.1:11434 > /dev/null; then
        echo "Ollama is UP and Running!"
        break
    fi
    sleep 5
    echo "Still waiting... ($i/20)"
done


# Example: DPR-BAG with Llama3.2:3b, BC prompt (no entity guidance).
# Adjust --model_name, --input_jsonl, --paragraph_prompt_version,
# --top_n_umls to run other configurations
echo "Running Python script..."
python pipeline.py \
    --model_name llama3.2:3b \
    --input_jsonl ./splitting/output/fs.jsonl \
    --paragraph_prompt_version bc \
    --top_n_umls 0

echo "Job finished. Killing Ollama server..."
kill $OLLAMA_PID