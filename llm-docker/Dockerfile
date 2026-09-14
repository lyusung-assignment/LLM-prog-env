# =====================================================================
#  LLM Research/Development GPU container (Ubuntu 22.04 + CUDA 12.6 + PyTorch 2.6 + HuggingFace + LangGraph + Jupyter)
#  Host(Windows) responsible: NVIDIA graphics driver + Docker Desktop(WSL2 backend)
#  Container responsible: CUDA toolkit / Anaconda / conda environment llm / Python stack
# =====================================================================
# CUDA devel image = nvcc included (flash-attn, bitsandbytes etc. for compilation)
# GPU driver is never installed -> host driver is used as-is

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

LABEL maintainer="llm-lab"
LABEL description="Python 3.11 + PyTorch(CUDA 12.6) + HuggingFace + PEFT/TRL + LangGraph + Jupyter"

# ---------- Default environment variables ----------
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/opt/conda \
    ENV_NAME=llm

ENV PATH=$CONDA_DIR/bin:$PATH

# ---------- OS packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        vim \
        htop \
        tmux \
        unzip \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        openssh-client \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# ---------- Miniconda installation ----------
# (Instead of the full Anaconda distribution, we use Miniconda + required packages only = image size reduction,
#  and avoidance of Anaconda commercial license issues. The conda usage experience is the same.)
ARG MINICONDA_VERSION=py311_24.7.1-0
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda config --system --set always_yes true \
    && $CONDA_DIR/bin/conda config --system --set channel_priority strict \
    && $CONDA_DIR/bin/conda clean -afy

# ---------- conda environment llm creation (Python 3.11) ----------
RUN conda create -n $ENV_NAME python=3.11 -y \
    && conda clean -afy

# Every RUN command after this will be executed in the llm conda environment
SHELL ["conda", "run", "--no-capture-output", "-n", "llm", "/bin/bash", "-lc"]

# ---------- PyTorch (CUDA 12.6 runtime included wheel) ----------
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu126

# ---------- HuggingFace stack ----------
RUN pip install \
        transformers==4.49.0 \
        datasets==3.3.2 \
        accelerate==1.4.0 \
        peft==0.14.0 \
        trl==0.15.2 \
        tokenizers \
        safetensors \
        sentencepiece \
        evaluate \
        huggingface_hub[cli] \
        bitsandbytes==0.45.3

# ---------- LangChain / LangGraph ----------
RUN pip install \
        langchain==0.3.9 \
        langchain-community==0.3.8 \
        langchain-core==0.3.21 \
        langchain-openai \
        langchain-huggingface \
        langgraph==0.2.53 \
        langsmith

# ---------- Jupyter + kernel registration + analysis tools ----------
RUN pip install \
        jupyterlab \
        notebook \
        ipykernel \
        ipywidgets \
        jupyterlab-widgets \
        pandas \
        numpy \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        tensorboard \
        tqdm \
        python-dotenv \
        rich \
    && python -m ipykernel install --sys-prefix --name llm --display-name "Python 3.11 (llm)"

# ---------- Container entrypoint: activate llm environment by default ----------
RUN conda init bash \
    && echo "conda activate ${ENV_NAME}" >> /root/.bashrc

SHELL ["/bin/bash", "-lc"]

ENV PATH=$CONDA_DIR/envs/llm/bin:$PATH \
    CONDA_DEFAULT_ENV=llm

# HuggingFace cache location
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    TORCH_HOME=/workspace/.cache/torch

WORKDIR /workspace
EXPOSE 8888

# ---------- Default execution: JupyterLab ----------
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.root_dir=/workspace"]
