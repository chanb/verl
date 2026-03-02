#!/bin/bash

cd /workspace/verl
pip3 install --no-deps -e .

if ! test -d /datasets/questa; then
  echo "generating questa dataset..."
  python3 /workspace/verl/examples/data_preprocess/questa.py --local_save_dir /datasets/questa
fi

echo "start training..."
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/datasets/questa/train.parquet \
 data.val_files=/datasets/questa/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 data.filter_overlong_prompts=True \
 actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 hydra.run.dir=/verl_checkpoints/openr1_math_220k \
 global_profiler.save_path=/verl_checkpoints/openr1_math_220k/outputs/ppo \
 trainer.project_name=openr1_math_220k \
 trainer.experiment_name=ppo \
 trainer.default_local_dir=/verl_checkpoints/openr1_math_220k/ppo \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15
