#!/bin/bash

# Build script for brain-image-implementation Singularity/Apptainer image

set -e


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TAG=${TAG:-"latest"}
DEFINITION_FILE=${DEFINITION_FILE:-"scripts/container/singularity.def"}
IMAGE_FILE=${IMAGE_FILE:-"images/brain_$(date +%Y_%m_%d_%H_%M_%S).sif"}
TMP_DIR=${APPTAINER_TMPDIR:-"/tmp"}

# Check if apptainer or singularity is available
if command -v apptainer &> /dev/null; then
    PROGRAM="apptainer"
elif command -v singularity &> /dev/null; then
    PROGRAM="singularity" 
else
    echo -e "${RED}Error: Neither apptainer nor singularity found in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Building ${PROGRAM} image: ${IMAGE_FILE} from definition: ${DEFINITION_FILE}...${NC}"

# Build the Singularity/Apptainer image
sudo -E ${PROGRAM} build --tmpdir ${TMP_DIR} \
    "${IMAGE_FILE}" \
    "${DEFINITION_FILE}"

echo -e "${GREEN}Singularity/Apptainer image built successfully!${NC}"
echo -e "${YELLOW}To open a shell in the container:${NC}"
echo -e "  ${PROGRAM} shell ${IMAGE_FILE}"
echo -e "${YELLOW}To run a script:${NC}"
echo -e "  ${PROGRAM} run ${IMAGE_FILE} YOUR_SCRIPT_HERE" 