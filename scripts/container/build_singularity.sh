#!/bin/bash

# Build script for brain-image-implementation Singularity/Apptainer image

set -e

# Configuration
DATETIME=$(date +%Y_%m_%d_%H_%M_%S)
IMAGE_NAME="brain_${DATETIME}"
TAG="latest"
DEFINITION_FILE="singularity.def"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Singularity/Apptainer image: ${IMAGE_NAME}.sif${NC}"

# Check if apptainer or singularity is available
if command -v apptainer &> /dev/null; then
    BUILD_CMD="apptainer build"
    echo -e "${YELLOW}Using Apptainer${NC}"
elif command -v singularity &> /dev/null; then
    BUILD_CMD="singularity build"
    echo -e "${YELLOW}Using Singularity${NC}"
else
    echo -e "${RED}Error: Neither apptainer nor singularity found in PATH${NC}"
    exit 1
fi

# Build the Singularity/Apptainer image
${BUILD_CMD} \
    --fakeroot \
    "${IMAGE_NAME}.sif" \
    "${DEFINITION_FILE}"

echo -e "${GREEN}Singularity/Apptainer image built successfully!${NC}"
echo -e "${YELLOW}To run the container:${NC}"
echo -e "  apptainer shell ${IMAGE_NAME}.sif"
echo -e "  # or"
echo -e "  singularity shell ${IMAGE_NAME}.sif"
echo -e "${YELLOW}To run a script:${NC}"
echo -e "  apptainer exec ${IMAGE_NAME}.sif python scripts/gen_embeddings.py" 