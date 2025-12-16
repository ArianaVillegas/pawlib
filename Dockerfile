# PAWlib Docker Image
# Production-ready container with GPU support

FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY examples/ ./examples/

# Install pawlib with all dependencies in single layer
RUN pip install --no-cache-dir -e ".[dev]" && \
    pip cache purge && \
    rm -rf /tmp/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["/bin/bash"]
