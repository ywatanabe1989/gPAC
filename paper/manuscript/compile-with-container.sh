#!/bin/bash
# Compile manuscript using the Apptainer container

set -e

# Check if container exists
if [ ! -f "latex-compiler.sif" ]; then
    echo "Container not found. Please run ./build-container.sh first"
    exit 1
fi

# Pass all arguments to the compile script inside the container
echo "Compiling manuscript with Apptainer container..."
apptainer exec latex-compiler.sif ./scripts/sh/compile.sh "$@"