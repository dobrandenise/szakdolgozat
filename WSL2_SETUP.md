WSL2 + CUDA 12.6 + RAPIDS quickstart (recommended for Windows users)

This guide shows how to set up WSL2 on Windows, install a conda environment for GPU-accelerated work (CUDA 12.6), and run quick smoke tests for PyTorch and RAPIDS (cudf/cuml).

IMPORTANT: RAPIDS packages are built against specific CUDA versions. The environment-gpu.yml in the repository pins cudatoolkit=12.6 per your request — if RAPIDS binaries for CUDA 12.6 are not yet available on the rapidsai channel, follow the RAPIDS installer instructions (https://rapids.ai/start) to pick a compatible RAPIDS release and CUDA. I include commands below that you can adapt to the exact RAPIDS version.

Prerequisites on Windows
- Windows 10/11 with WSL2 support
- Administrative privileges to enable features and install drivers
- NVIDIA GPU with drivers supporting CUDA on WSL (your driver 581.57 is recent)

1) Enable WSL2 & install a Linux distro
- Open PowerShell as Administrator and run:
  wsl --install -d Ubuntu-22.04
  (If WSL is already installed, ensure it's WSL2: wsl --set-default-version 2)
- Restart if prompted. Launch Ubuntu from Start menu and complete initial setup.

2) Install NVIDIA "CUDA on WSL" driver on Windows
- Download and install the NVIDIA Windows driver for CUDA on WSL (see NVIDIA docs): https://developer.nvidia.com/cuda/wsl
- Reboot Windows if required.

3) In WSL (Ubuntu) verify GPU access
- In the WSL shell:
  sudo apt update && sudo apt upgrade -y
  # Optional: install common build tools
  sudo apt install -y build-essential git wget ca-certificates
  # Check driver visibility
  nvidia-smi
  # If nvidia-smi runs and shows GPU, you're ready to proceed.

4) Install Miniconda and mamba in WSL
- Download & install Miniconda (Linux) inside WSL:
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  eval "$($HOME/miniconda/bin/conda shell.bash hook)"
  conda init
  source ~/.bashrc
- Install mamba (faster conda):
  conda install -n base -c conda-forge mamba

5) Create the base GPU environment
- Option A: Use the repository environment file (may require manual RAPIDS install afterwards):
  mamba env create -f /path/to/repo/environment-gpu.yml
  conda activate fraud-dnn-gpu

- Option B (recommended): Install RAPIDS + PyTorch with explicit RAPIDS version matching CUDA 12.6. Replace <RAPIDS_VER> with a supported RAPIDS release (use https://rapids.ai/start to find the exact release for your CUDA):
  # Example (replace RAPIDS_VER with the desired version like 23.12 if it supports CUDA 12.6):
  mamba create -n fraud-dnn -c rapidsai -c nvidia -c conda-forge \
    rapids=<RAPIDS_VER> cudatoolkit=12.6 python=3.10
  conda activate fraud-dnn

  # Then install PyTorch (CUDA 12.6 build) via NVIDIA/pytorch channels (example):
  mamba install -c pytorch -c nvidia pytorch pytorch-cuda=12.6 torchvision -y

  # Finally, install the Python pip-only extras:
  pip install optuna==4.9.0 category_encoders==2.6.0 shap==0.49.1 \
    torchmetrics==0.11.0 pytorch-lightning==2.0.0 mlflow==2.4.0 pandas-profiling==3.7.0

Notes on selecting RAPIDS version
- Visit https://rapids.ai/start and pick the RAPIDS release that lists support for CUDA 12.6 (if available). If CUDA 12.6 is not listed, choose the nearest supported CUDA (12.4/12.5/12.1 etc.), and set cudatoolkit accordingly in the conda create command.
- To list available RAPIDS builds from your machine:
  conda search -c rapidsai rapids --info

6) Quick smoke tests
- In WSL inside the activated conda env:
  python -c "import torch; print('torch.cuda.available', torch.cuda.is_available(), 'torch cuda version', torch.version.cuda)"
  python -c "import cudf; print('cudf version', cudf.__version__)"
  python -c "import cuml; print('cuml version', cuml.__version__)"

7) Memory & workflow tips for 6 GB VRAM
- Use float32, not float64. Convert large numeric arrays to float32 to save VRAM.
- Avoid copying the whole dataset to GPU at once. Use dask-cudf to process in partitions or convert only the columns you need to GPU.
- When training on GPU, reduce batch size (e.g., 32 -> 8 -> 4) until it fits; use gradient accumulation to simulate larger batch sizes.
- Use mixed precision (AMP) in PyTorch to reduce memory and speed up training when supported.
- For expensive algorithms (HDBSCAN), consider running on sampled subsets or use CPU-run as fallback.

8) Troubleshooting
- If cudf import fails: ensure the RAPIDS version matches the cudatoolkit and that the rapidsai channel has a build for your CUDA.
- If conda solver hangs: use mamba instead of conda.
- If GPU memory OOM: reduce batch size, use cudf/dask partitions, or run heavy steps on CPU / remote GPU.

9) Example workflow for this project
- Do feature engineering and correctness checks on CPU (safe and fast turnaround).
- For heavy aggregation, clustering, and final training runs, use WSL2+RAPIDS+PyTorch GPU. Start with small samples to tune memory parameters, then scale.

If you want, I will:
- Update environment-gpu.yml in the repo to cudatoolkit=12.6 (done) and push a PR (I can open the PR for you).  
- Check the rapidsai channel for the RAPIDS release that supports CUDA 12.6 and update the recommended conda create command with the exact release number.  

Tell me if you want me to detect the exact RAPIDS release available for CUDA 12.6 and update the instructions with a concrete release string.