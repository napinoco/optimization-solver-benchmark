# Optimization Solver Benchmark System - Docker Environment
# Multi-stage build for Python + MATLAB Runtime integration

#==============================================================================
# Stage 1: Python Dependencies and System Tools
#==============================================================================
FROM python:3.12-slim AS python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    git \
    procps \
    vim \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Detect architecture for SCIP installation strategy
ARG TARGETARCH
RUN echo "Building for architecture: ${TARGETARCH:-unknown}"

# Install SCIP based on architecture (ARM64 needs build from source, x86_64 can use pip)
ARG INCLUDE_SCIP=true

# For ARM64: Install SCIP from source if requested
RUN if [ "$INCLUDE_SCIP" = "true" ] && [ "${TARGETARCH}" = "arm64" -o "${TARGETARCH}" = "aarch64" ]; then \
        echo "ARM64 detected: Installing SCIP build dependencies..." && \
        apt-get update && apt-get install -y \
            cmake \
            libgmp-dev \
            libboost-dev \
            libboost-program-options-dev \
            libboost-filesystem-dev \
            libboost-regex-dev \
            libboost-thread-dev && \
        cd /tmp && \
        wget -q https://github.com/scipopt/scip/archive/refs/tags/v920.tar.gz && \
        tar xf v920.tar.gz && \
        cd scip-920 && \
        mkdir build && cd build && \
        cmake -DCMAKE_BUILD_TYPE=Release .. && \
        make -j$(nproc) && \
        make install && \
        cd / && rm -rf /tmp/scip-920 /tmp/v920.tar.gz && \
        rm -rf /var/lib/apt/lists/* && \
        echo "SCIP installed from source for ARM64"; \
    fi

# Create non-root user for security
RUN groupadd -r solver && useradd -r -g solver -d /home/solver -s /bin/bash solver \
    && mkdir -p /home/solver && chown -R solver:solver /home/solver

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
COPY requirements-no-scip.txt /tmp/requirements-no-scip.txt

# Install dependencies - architecture-aware for SCIP
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$INCLUDE_SCIP" = "true" ]; then \
        if [ "${TARGETARCH}" = "arm64" -o "${TARGETARCH}" = "aarch64" ]; then \
            echo "ARM64: Installing pyscipopt from source with existing SCIP library..." && \
            pip install --no-cache-dir -r /tmp/requirements.txt; \
        else \
            echo "x86_64: Installing pyscipopt from pre-built wheel..." && \
            pip install --no-cache-dir -r /tmp/requirements.txt; \
        fi \
    else \
        echo "SCIP disabled: Installing without pyscipopt..." && \
        pip install --no-cache-dir -r /tmp/requirements-no-scip.txt; \
    fi

#==============================================================================
# Stage 2: Octave Environment (MATLAB compatibility layer)
#==============================================================================
# Note: Using Octave instead of MATLAB for Docker compatibility
# MATLAB licenses do not support virtual environments like Docker

#==============================================================================
# Stage 3: Final Production Environment
#==============================================================================
FROM ubuntu:22.04 AS production

# Set non-interactive frontend to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies including Octave and MEX compilation tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    procps \
    unzip \
    libgomp1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    # Python dependencies for copying from python-base
    libffi8 \
    libssl3 \
    libbz2-1.0 \
    libreadline8 \
    libsqlite3-0 \
    libncursesw6 \
    libgdbm6 \
    liblzma5 \
    tk \
    # Octave for MATLAB compatibility with MEX compilation support
    octave \
    liboctave-dev \
    # Build tools for MEX compilation
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python 3.12 from python-base stage
COPY --from=python-base /usr/local /usr/local

# Ensure Python and pip are in PATH
ENV PATH=/usr/local/bin:$PATH

# Verify Python installation
RUN python3 --version && pip3 --version

# Create non-root user with same UID/GID as host user
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create group (handle existing GID gracefully)
RUN (getent group ${GROUP_ID} || groupadd -r solver -g ${GROUP_ID}) && \
    useradd -r -u ${USER_ID} -g ${GROUP_ID} -d /home/solver -s /bin/bash solver && \
    mkdir -p /home/solver && \
    chown -R solver:${GROUP_ID} /home/solver

# Octave environment setup (MATLAB compatibility)
# No license restrictions for Octave
ENV OCTAVE_PATH="/usr/share/octave/packages"

# Set up application directory
WORKDIR /app
RUN chown solver:${GROUP_ID} /app

# Copy application code (will be overridden by volume mount in development)
COPY --chown=solver:${GROUP_ID} . /app/

# Switch to non-root user
USER solver

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# For MATLAB/Octave headless execution
ENV DISPLAY=""
ENV _JAVA_OPTIONS="-Djava.awt.headless=true"
ENV MATLAB_LOG_DIR="/tmp"

# Default working directory for solver execution
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import sys; print('Python OK'); sys.exit(0)" || exit 1

# Entry point script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
USER root
RUN chmod +x /usr/local/bin/entrypoint.sh
USER solver

# Default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py", "--help"]