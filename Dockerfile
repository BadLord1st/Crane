FROM nvidia/cuda:13.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	curl \
	git \
	pkg-config \
	libssl-dev \
	clang \
	cmake \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /work

# Copy full workspace and build CUDA-enabled binary.
COPY . .
ARG HOST_OS=unknown
ARG CUDA_COMPUTE_CAP
RUN if [ "${HOST_OS}" = "darwin" ]; then \
	echo "ERROR: CUDA Docker image build from macOS host is blocked for safety. Build on Linux/NVIDIA host or pass --build-arg HOST_OS=linux if intentional." >&2; \
	exit 1; \
	fi
RUN if [ -z "${CUDA_COMPUTE_CAP}" ]; then \
	echo "ERROR: CUDA_COMPUTE_CAP is required (e.g. 90 for H100, 89 for RTX 4070)." >&2; \
	exit 1; \
	fi
ENV CUDA_HOME=/usr/local/cuda
ENV CUDARC_CUDA_VERSION=13010
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
RUN cargo build -p crane-oai --release --features cuda

# Minimal runtime image with only resulting binary.
FROM nvidia/cuda:13.1.1-runtime-ubuntu22.04 AS runtime
WORKDIR /app
COPY --from=builder /work/target/release/crane-oai /app/crane-oai
ENTRYPOINT ["/app/crane-oai"]
