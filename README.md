# ML Mamba — Environment Setup & HPC Workflow

## Table of Contents

- [What is Mamba?](#what-is-mamba)
- [Mamba vs Transformer](#mamba-vs-transformer)
- [Key Concepts](#key-concepts)
- [Environment Setup](#environment-setup)
- [HPC Cluster Commands](#hpc-cluster-commands)
- [Model Download (Hugging Face)](#model-download-hugging-face)
- [Package Installation](#package-installation)
- [File Transfer & Copy](#file-transfer--copy)

---

## What is Mamba?

**Mamba** is a novel deep learning architecture based on **Selective State Space Models (SSMs)**. Introduced in 2023 by Gu & Dao, it is designed as a competitive alternative to Transformers for sequence modeling tasks, offering **linear-time complexity** instead of the quadratic complexity of self-attention.

Mamba is particularly strong in:
- Long-sequence modeling (text, audio, genomics, time series)
- Memory-efficient inference
- Tasks where context length is very large

> Paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

---

## Mamba vs Transformer

| Feature | Transformer | Mamba (SSM) |
|---|---|---|
| Complexity | O(n²) — quadratic | O(n) — linear |
| Attention mechanism | Self-attention | Selective state space |
| Long-sequence scaling | Poor | Strong |
| Inference memory | High (KV cache) | Low (fixed-size state) |
| Parallelism (training) | Full | Efficient via scan |
| Best use case | NLP, vision (shorter seq.) | Long sequences, streaming |

---

## Key Concepts

### State Space Models (SSMs)

SSMs map an input sequence `x(t)` to an output `y(t)` through a hidden state `h(t)`:

```
h'(t) = A·h(t) + B·x(t)
 y(t) = C·h(t)
```

Where `A`, `B`, `C` are learned parameters. Mamba makes these **input-dependent** (selective), enabling the model to filter irrelevant information dynamically.

### Selective Scan (S6)

The core of Mamba is the **S6 block**, where the SSM parameters (`B`, `C`, `Δ`) depend on the input, allowing content-aware state updates — something classical SSMs cannot do.

### Mamba Block Architecture

```
Input
  │
  ├──────────────────────────────┐
  │                              │
Linear Projection           Linear Projection
  │                              │
SSM (Selective Scan)         SiLU Activation
  │                              │
  └──────────── × ───────────────┘
                │
          Linear Projection
                │
             Output
```

### Mamba-2

Mamba-2 (2024) introduces **State Space Duality (SSD)**, connecting SSMs to a new form of structured attention, enabling better scaling and hybrid architectures (e.g., Jamba, Zamba).

---

## Environment Setup

### Using Conda

```bash
conda create -p env python=3.12 -y
conda activate ./env
```

### Using Mamba (faster solver)

```bash
mamba create -p env python=3.12 -y
mamba activate ./env
```

> `-p env` creates the environment in a local `./env` directory instead of the central conda store — useful on HPC systems with quota limits.

---

## HPC Cluster Commands

### Check Account Balance & Quota

```bash
sbalance       # Show remaining compute allocation (SUs/hours)
myquota        # Show storage quota usage
```

### Monitor Jobs

```bash
squeue         # List all jobs in the queue
squeue --me    # List only your jobs
```

### Submit & Cancel Jobs

```bash
sbatch submit.sh          # Submit a batch job script
scancel {job_id}          # Cancel a specific job by ID
```

### Example `submit.sh`

```bash
#!/bin/bash
#SBATCH --job-name=mamba_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

source activate ./env

python train.py
```

---

## Model Download (Hugging Face)

### Install Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
```

### Authenticate

```bash
hf auth login
```

> You will be prompted for a Hugging Face token. Generate one at https://huggingface.co/settings/tokens

### Download a Model

```bash
hf download amazon/chronos-2 --local-dir ./model
```

> Downloads the **Amazon Chronos-2** time series forecasting model (based on language model architecture) into a local `./model` directory.

---

## Package Installation

### PyArrow support for Pandas

```bash
pip install 'pandas[pyarrow]'
```

### Chronos Forecasting

```bash
pip install --upgrade chronos-forecasting
```

### PyTorch with CUDA 11.8

```bash
pip install --default-timeout=1000 \
    torch==2.5.1+cu118 \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

> `--default-timeout=1000` prevents timeout errors when downloading large packages on slow or shared network connections.

#### CUDA Version Reference

| CUDA | `--extra-index-url` suffix |
|---|---|
| 11.8 | `cu118` |
| 12.1 | `cu121` |
| 12.4 | `cu124` |
| CPU only | `cpu` |

---

## File Transfer & Copy

### Download from URL

```bash
wget <url>
wget -c <url>              # Resume interrupted download
wget -O output_name <url>  # Save with a custom filename
```

### Copy Files

```bash
cp source destination
cp -r source_dir/ dest_dir/    # Recursive copy for directories
cp -p source destination       # Preserve timestamps and permissions
```

---

## Quick Reference

```bash
# 1. Create and activate environment
mamba create -p env python=3.12 -y && mamba activate ./env

# 2. Install dependencies
pip install -U "huggingface_hub[cli]" 'pandas[pyarrow]' chronos-forecasting
pip install --default-timeout=1000 torch==2.5.1+cu118 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Download model
hf auth login
hf download amazon/chronos-2 --local-dir ./model

# 4. Submit job
sbatch submit.sh
squeue --me

# 5. Cancel if needed
scancel {job_id}
```
