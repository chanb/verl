# Setup apptainer
```
apptainer build verl.sif docker://verlai/verl:vllm015.dev
```

# Clone Verl
```
git clone git@github.com:chanb/verl.git
```

# Start interactive job
```
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
module load apptainer/1.4.5
```

# Start apptainer
```
cd research/hint_without_realizability
apptainer run --nv -C -W $SLURM_TMPDIR -B .:/workspace -B ~/scratch/datasets:/datasets verl.sif
cd /workspace/verl
pip install --no-deps -e .
pip install ipdb
```

# Download dataset (one-time only)
```
# python3 examples/data_preprocess/gsm8k.py --local_save_dir /datasets/gsm8k
python3 examples/data_preprocess/questa.py --local_save_dir /datasets/questa
```

# Download model
```
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```

# Run PPO
```
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/datasets/questa/train.parquet \
 data.val_files=/datasets/questa/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 data.filter_overlong_prompts=True \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
```