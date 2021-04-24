ARG PYTORCH="1.6.0"

FROM pytorch/pytorch

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update 
RUN apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMCV
# RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
# ENV FORCE_CUDA="1"
RUN pip install mmcv-full==1.2.0+torch1.6.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html
RUN pip install numpy==1.20.0
RUN pip install poseval@git+https://github.com/svenkreiss/poseval.git

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN mkdir -p /mmpose/data
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Flask
RUN pip install flask flask_cors