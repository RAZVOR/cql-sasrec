FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python, dependencies for mujoco-py, from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    build-essential \
    patchelf \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Jupyter
RUN pip3 install --no-cache-dir jupyter

# Req
COPY requirements.txt requirements.txt
RUN pip install --progress-bar=on --no-cache-dir -r requirements.txt

# Create working directory and set it as default
WORKDIR /workspace

# Mount current directory to Jupyter notebooks directory
VOLUME . /workspace

# Expose port for Jupyter
EXPOSE 8886

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8886", "--no-browser", "--allow-root", "--notebook-dir=/workspace", "--NotebookApp.token='1234'"]