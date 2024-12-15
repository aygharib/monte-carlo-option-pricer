# Base image
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install basic tools and xmake
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3 \
    python3-pip \
    && curl -fsSL https://xmake.io/shget.text | bash \
    && rm -rf /var/lib/apt/lists/*

# Add xmake to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Install Conan via pip
RUN pip3 install conan --break-system-packages

# Set the default working directory
WORKDIR /workspace

# Default command to run
CMD ["/bin/bash"]
