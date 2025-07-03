#!/bin/bash

# Build script for brain-image-implementation Docker image

set -e

# Configuration
IMAGE_NAME="brain-image-implementation"
TAG="latest"
DOCKERFILE="Dockerfile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Docker image: ${IMAGE_NAME}:${TAG}${NC}"

# Build the Docker image
docker build \
    --tag "${IMAGE_NAME}:${TAG}" \
    --file "${DOCKERFILE}" \
    --progress=plain \
    .

echo -e "${GREEN}Docker image built successfully!${NC}"
echo -e "${YELLOW}To run the container:${NC}"
echo -e "  docker run -it --rm ${IMAGE_NAME}:${TAG}"
echo -e "${YELLOW}To convert to Singularity/Apptainer:${NC}"
echo -e "  apptainer build brain-image-implementation.sif docker://${IMAGE_NAME}:${TAG}" 