# image
FROM python:3.11-bullseye

# System libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python libraries
RUN pip install --upgrade pip && pip install --no-cache-dir \
    tensorflow==2.16.1 \
    keras \
    scikit-learn \
    opencv-contrib-python \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    jupyterlab \
    ultralytics

# Working directory
WORKDIR /deeplearning

# Command to run
CMD ["bash"]
