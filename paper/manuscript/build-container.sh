#!/bin/bash
# Build the LaTeX compilation container

set -e

echo "Building LaTeX compilation container..."
echo "This may take a while as it downloads and installs TeX Live..."

# Build the container
apptainer build latex-compiler.sif latex-compiler.def

if [ $? -eq 0 ]; then
    echo "Container built successfully: latex-compiler.sif"
    echo "You can now use compile-with-container.sh to compile the manuscript"
else
    echo "Error building container"
    exit 1
fi