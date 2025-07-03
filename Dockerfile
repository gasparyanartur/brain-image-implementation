# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy pyproject.toml and other config files first for better caching
COPY pyproject.toml README.md ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install Python dependencies using uv
RUN uv pip install -e .

# Create a non-root user for running the application
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Set the default command
CMD ["bash"] 