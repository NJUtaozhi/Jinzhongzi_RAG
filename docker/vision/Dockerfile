# OpenFace 2.2.0 officially targets the Ubuntu 18.04/20.04 generation.
# Keep the build pinned so a future upstream change cannot silently break AU output.
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENFACE_REF=OpenFace_2.2.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libboost-filesystem-dev \
    libboost-system-dev \
    libdlib-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenblas-dev \
    libopencv-dev \
    libpng-dev \
    libtbb-dev \
    libtiff-dev \
    pkg-config \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch "${OPENFACE_REF}" \
    https://github.com/TadasBaltrusaitis/OpenFace.git OpenFace

# CE-CLM patch experts are not stored in Git because of their size.
WORKDIR /opt/OpenFace
RUN bash ./download_models.sh
RUN cmake -S . -B build \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_CXX_STANDARD=17 \
    && cmake --build build --target FaceLandmarkImg --parallel 2 \
    && test -x /opt/OpenFace/build/bin/FaceLandmarkImg \
    && test -d /opt/OpenFace/build/bin/model

ENV OPENFACE_BIN=/opt/OpenFace/build/bin/FaceLandmarkImg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY docker/vision/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

# 复制服务代码
COPY vision_service/ /app/

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
