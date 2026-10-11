# syntax=docker/dockerfile:1
#
# Two targets:
#   python - the Python SDK + CLI (used by the `loadgen` Compose service)
#   cpp    - the native library and demo, built against opentelemetry-cpp from source
#
#   docker build --target python -t inference-profiler:python .
#   docker build --target cpp    -t inference-profiler:cpp .

# ------------------------------------------------------------------- python
FROM python:3.12-slim-bookworm AS python
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
RUN python -m pip install --upgrade pip && python -m pip install .
RUN useradd --create-home --uid 10001 profiler
USER profiler
ENTRYPOINT []
CMD ["inference-profiler", "--help"]

# ---------------------------------------------------------------- cpp build
FROM ubuntu:24.04 AS cpp-build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates \
      libcurl4-openssl-dev libprotobuf-dev protobuf-compiler zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt CMakePresets.json ./
COPY cpp ./cpp
COPY tests/cpp ./tests/cpp
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DIPP_FETCH_OPENTELEMETRY=ON \
    && cmake --build build \
    && ctest --test-dir build --output-on-failure

# ---------------------------------------------------------------------- cpp
FROM ubuntu:24.04 AS cpp
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4t64 libprotobuf32t64 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cpp-build /src/build/cpp_inference_demo /usr/local/bin/cpp_inference_demo
RUN useradd --create-home --uid 10001 profiler
USER profiler
CMD ["cpp_inference_demo", "--help"]
