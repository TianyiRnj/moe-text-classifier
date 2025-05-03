# README: Running `train.py` and `inference.py` on PSC

This guide explains how to launch an interactive GPU job on PSC and run our two scripts:

- `train.py` — Accelerate-powered train and evaluation
- `inference.py` — DeepSpeed-powered end-to-end evaluation

Note: Replace <project_account> and <PROJECT_DIR> with your own information.

---

1. Interactive Session Allocation

---

Each PSC node can have up to 8 NVIDIA GPUs (V100-16GB, V100-32GB, or H100-80GB).  
To request an interactive job on 8×H100-80 GPUs with 8-hour wall time:

    interact -p GPU -t 08:00:00 --gres=gpu:h100-80:8 -A <project_account>

---

2. Clean & Configure Cache Directories

---

Clear stale caches and redirect them to your project storage:

    rm -rf ~/.cache/huggingface ~/.cache/torch ~/.cache/triton
    rm -rf ~/.cache/torch_extensions/*

    export PROJECT_DIR=/path/to/your/project
    export TRANSFORMERS_CACHE=$PROJECT_DIR/hf_cache
    export HF_HOME=$TRANSFORMERS_CACHE
    export HF_MODULES_CACHE=$TRANSFORMERS_CACHE
    export TRITON_CACHE_DIR=$PROJECT_DIR/triton_cache
    export TORCH_EXTENSIONS_DIR=$PROJECT_DIR/torch_ext

    mkdir -p $TRANSFORMERS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR

---

3. Load Modules and Activate Your Environment

---

Load necessary modules and activate your environment:

    module load cuda/11.8
    module load gcc/9.3
    module load python/3.8

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate moe-env

Check:

    which python
    python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"
    deepspeed --version

---

4. Run `train.py` for Custom MoE Inference Latency

---

Syntax:

    accelerate launch --num_processes 4 /train.py 2>&1 | tee test_dp${DP}_ep${EP}.log

---

5. Run `inference.py` with DeepSpeed

---

Step 1: Edit your `inference_config.json`:

{
"kernel_inject": true,
"dtype": "fp16",
"tensor_parallel": { "tp_size": 1 },
"moe": {
"enabled": true,
"ep_size": <EP>,
"moe_experts": [8]
},
"enable_cuda_graph": false
}

Step 2: Launch with DeepSpeed

    NUM_GPUS=8
    deepspeed --num_gpus=${NUM_GPUS} inference.py \\
      --config inference_config.json \\
      --model_path $PROJECT_DIR/model \\
      --data_path  $PROJECT_DIR/data.csv \\
      2>&1 | tee ds_inference_${NUM_GPUS}_ep${EP}.log

Step 3: Interpretation

You will see:

    [Rank 0] F1 score: 0.8421
    [Rank 0] Time    : 45.2s
    [Rank 0] Throughput: 175.6 samples/s

Repeat with different `num_gpus` and `ep_size` to explore performance.

---

6. Extra Tips

---

- If OOM, reduce EP or batch size.
- Always use `tee` to save logs.
- Clean cache directories before reruns.
- Use the same conda environment for both scripts.
- Log throughput and latency per config in a table.
