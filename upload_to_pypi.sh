#!/bin/bash
# Script to upload gPAC to PyPI

echo "========================================="
echo "gPAC v0.2.0 - PyPI Upload Script"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run from the gPAC root directory."
    exit 1
fi

echo ""
echo "Pre-upload checklist:"
echo "✓ Version updated to 0.2.0 in src/gpac/__init__.py"
echo "✓ CHANGELOG.md updated with v0.2.0 changes"
echo "✓ Major fix: Unbiased surrogate generation"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ src/*.egg-info

# Build the package
echo "Building package..."
python -m build

# Check if build was successful
if [ ! -d "dist" ]; then
    echo "Error: Build failed. No dist directory created."
    exit 1
fi

echo ""
echo "Build successful! Files created:"
ls -la dist/

echo ""
echo "To upload to TestPyPI (recommended first):"
echo "  python -m twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI:"
echo "  python -m twine upload dist/*"
echo ""
echo "Note: You'll need to have your PyPI credentials configured."
echo "You can use ~/.pypirc or provide them when prompted."
echo ""
echo "After upload, users can install with:"
echo "  pip install gpac==0.2.0"