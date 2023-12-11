# Ubuntu 20.04, CUDA 12.0.1, cuDNN 8.0, TensorRT 8.5.2.2
FROM nvcr.io/nvidia/tensorrt:23.01-py3
ARG OPENCV_VERSION="4.8.0"
ARG DEBIAN_FRONTEND=noninteractive
LABEL maintainer="ht.nah98@gmail.com"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ git wget cmake sudo \
    build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libcanberra-gtk-module libcanberra-gtk3-module

# Install Python3 pip and cmake
RUN apt-get install -y python3-pip && \
    pip3 install cmake

# Setting working directory
WORKDIR /app

# Copy the entire content of your local directory to /app in the Docker image
COPY . /app

# Clone OpenCV repository
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git

# Build OpenCV
RUN cd opencv && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Clone Ultralytics repository
RUN git clone https://github.com/ultralytics/ultralytics

# Install Ultralytics package in editable mode
RUN pip install cython && \
    cd /app/ultralytics && \
    pip install -e .

# Specify the command to run your executable with arguments
CMD ["./build/detect_object_video", "--model", "/app/models/morning.onnx", "--input", "/app/images/morning_12.mp4"]
