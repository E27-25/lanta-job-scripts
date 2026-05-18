# ⚡ ML Mamba — Environment Setup & HPC Workflow

> **Platform:** LANTA HPC · **Stack:** Conda / Mamba · **GPU:** CUDA 11.8+

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [🛠️ Environment Setup](#️-environment-setup) |
| 2 | [🖥️ HPC Cluster Commands](#️-hpc-cluster-commands) |
| 3 | [🤗 Model Download (Hugging Face)](#-model-download-hugging-face) |
| 4 | [📦 Package Installation](#-package-installation) |
| 5 | [📁 File Transfer & Copy](#-file-transfer--copy) |
| 6 | [⚡ Quick Reference](#-quick-reference) |

---

## 🛠️ Environment Setup

### Step 1 — Load Mamba module

> ⚠️ **Required** — must be run every session (login or job script) before using `conda` or `mamba`.

```bash
ml Mamba
```

`ml` is shorthand for `module load`.

---

### Step 2 — Create & activate environment

#### 🐍 Using Conda

```bash
conda create -p env python=3.12 -y
conda activate ./env
```

#### 🐍 Using Mamba *(faster solver — recommended)*

```bash
mamba create -p env python=3.12 -y
mamba activate ./env
```

> 💡 `-p env` creates the environment in a local `./env` directory instead of the central conda store — useful on HPC systems with quota limits.

---

## 🖥️ HPC Cluster Commands

### 💰 Check Account Balance & Quota

```bash
sbalance    # Show remaining compute allocation (SUs/hours)
myquota     # Show storage quota usage
```

### 👀 Monitor Jobs

```bash
squeue          # List all jobs in the queue
squeue --me     # List only your jobs
```

### 🚀 Submit & Cancel Jobs

```bash
sbatch submit.sh     # Submit a batch job script
scancel {job_id}     # Cancel a specific job by ID
```

---

### 📄 Example `submit.sh`

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

## 🤗 Model Download (Hugging Face)

### 1 — Install Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
```

### 2 — Authenticate

```bash
hf auth login
```

> 🔑 Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3 — Download a Model

```bash
hf download amazon/chronos-2 --local-dir ./model
```

> Downloads **Amazon Chronos-2** (time series forecasting model) into `./model`

---

## 📦 Package Installation

### PyArrow + Pandas

```bash
pip install 'pandas[pyarrow]'
```

### Chronos Forecasting

```bash
pip install --upgrade chronos-forecasting
```

### PyTorch with CUDA

```bash
pip install --default-timeout=1000 \
    torch==2.5.1+cu118 \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

> ⏱️ `--default-timeout=1000` prevents timeouts when downloading large packages on slow or shared networks.

#### 🎯 CUDA Version Reference

| CUDA Version | Index URL Suffix |
|:---:|:---:|
| **11.8** | `cu118` |
| **12.1** | `cu121` |
| **12.4** | `cu124` |
| CPU only | `cpu` |

---

## 📁 File Transfer & Copy

### ⬇️ Download from URL

```bash
wget <url>
wget -c <url>               # Resume interrupted download
wget -O output_name <url>   # Save with a custom filename
```

### 📋 Copy Files

```bash
cp source destination
cp -r source_dir/ dest_dir/     # Recursive copy for directories
cp -p source destination        # Preserve timestamps and permissions
```

---

## ⚡ Quick Reference

> Full setup from scratch — copy & paste ready.

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load module (required before conda/mamba)
ml Mamba

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Create and activate environment
mamba create -p env python=3.12 -y
mamba activate ./env

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Install dependencies
pip install -U "huggingface_hub[cli]" 'pandas[pyarrow]' chronos-forecasting
pip install --default-timeout=1000 torch==2.5.1+cu118 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Download model
hf auth login
hf download amazon/chronos-2 --local-dir ./model

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Submit job
sbatch submit.sh
squeue --me

# 6. Cancel if needed
scancel {job_id}
```

---

<div align="center">

*LANTA HPC · Updated 2025*

</div>
