#!/bin/bash

# Job configuration
#SBATCH --job-name=qwen2_5_vl_3b_sft_mimi_cxr 
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=600GB
#SBATCH --cpus-per-task=16
#SBATCH --partition=prod_long

#SBATCH --error=/dev/null
#SBATCH --output=/dev/null

###SBATCH --output=editgrpo.out
###SBATCH --error=editgrpo.err

# set -x  # Enable debug mode

# Initialize your conda environment here
# conda init
eval "$(conda shell.bash hook)"
conda activate editgrpo  # Replace with your actual environment name

# Avoid error with limit for number of open file descriptors
ulimit -n 4096

# Set environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH=/mnt/ckpt/sft-mimic-cxr-3b-ep2/checkpoint  # replace it with your local file path
# export WANDB_DISABLED=true  # Disable wandb
export WANDB_MODE=offline

# Set master address and port
if [ -z $MASTER_ADDR ]; then
    echo "MASTER_ADDR is empty"
    export MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi
if [ -z $MASTER_PORT ]; then
    echo "MASTER_PORT is empty"
    export MASTER_PORT=29999
fi

# Calculate GPU and worker settings
NGPU_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c "$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')")
GPU=$((${NGPU_PER_NODE}))
WORKERS=$((${NGPU_PER_NODE} * 4))

if [ $WORKERS -gt 112 ]; then
    WORKERS=112
fi

echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
echo GPU=${GPU}
echo WORKERS=$WORKERS
# Set Ray address for all subsequent commands
export RAY_ADDRESS=$MASTER_ADDR:$MASTER_PORT

# Kill all Ray processes more aggressively
# pkill -9 -f ray || true

# Clean up any existing Ray processes
echo "Stopping any existing Ray processes..."
ray stop -f || true
sleep 5

# Create user-specific temp directory
mkdir -p $HOME/ray_tmp

# Start Ray head node
echo "Starting new Ray head node..."
ray start --head \
    --port=$MASTER_PORT \
    --node-ip-address=$MASTER_ADDR \
    --num-gpus=${GPU} \
    --temp-dir=$HOME/ray_tmp \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=9999

# Verify Ray is running
echo "Verifying Ray status..."
ray status --address=$MASTER_ADDR:$MASTER_PORT || { echo "Failed to start Ray cluster"; exit 1; }

# Wait for Ray to fully initialize
sleep 10

# Run the training script
python3 -m verl.trainer.main \
    config=examples/editgrpo_mimic_cxr_2_5.yaml \
    data.train_files=examples/data/sample_3000_mimic_cxr.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=True \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.extractor=True \
    worker.rollout.extractor_server='http://EXTRACTOR_SERVER_HOST:5001/analyze' \
    worker.rollout.extractor_num_ops_per_para=1 \
    worker.rollout.extractor_num_para=-1 \
    worker.rollout.extractor_type="ratescore" \
    worker.rollout.extractor_similarity_threshold=0.6 \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.reward.compute_score=rad_server \
    worker.reward.reward_server='http://REWARD_SERVER_HOST:5000/predict' \
    worker.reward.type='free_form' \
    worker.reward.reward_components='["radgraphf1", "cxb14micro", "ratescore"]' \
    trainer.experiment_name=qwen2_5_vl_3b_sft_mimic_cxr_threshold_0_6 \
    trainer.n_gpus_per_node=${GPU} \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=/mnt/ckpt/ \
    trainer.val_freq=-1 \
    trainer.save_freq=10 \
    trainer.save_limit=1 \
    algorithm.force_gt_num=0 \
    algorithm.loss_agg_mode="seq-mean-token-sum-norm" \
    algorithm.norm_adv_by_std_in_grpo=False \


    # worker.reward.reward_components=["radgraphf1", "cxb14micro", "format", "cxb_bal_acc", "ratescore"] \
    # trainer.load_checkpoint_path=qwen2_5_vl_3b_sft_mimic_cxr_threshold_0_6/global_step_xx \
