#!/bin/bash

export WANDB_API_KEY="09f781a401d2ec1d3ff5f9b550eb1ec0f291fb31"

export HOME=/home/tianbing_xu/repos/GRPO/TinyZero
echo "HOME: $HOME"

# Configuration
export N_GPUS=4
export BASE_MODEL=Qwen/Qwen2.5-1.5B
export DATA_DIR=$HOME/data/gsm8k
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=qwen2_1_5b_math
export VLLM_ATTENTION_BACKEND=XFORMERS

# Main script execution
echo "Starting training with the following configuration:"
echo "Number of GPUs: $N_GPUS"
echo "Base Model: $BASE_MODEL"
echo "Data Directory: $DATA_DIR"
echo "Rollout TP Size: $ROLLOUT_TP_SIZE"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Attention Backend: $VLLM_ATTENTION_BACKEND"


export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=Reinfoce \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee verl_demo_grpo_3_27.log