#!/bin/bash
# Upload gPAC v0.3.2 to PyPI

echo "Uploading gPAC v0.3.2 to PyPI..."
echo ""
echo "To upload to TestPyPI first (recommended):"
echo "python -m twine upload --repository testpypi dist/gpu_pac-0.3.2*"
echo ""
echo "To upload to PyPI:"
echo "python -m twine upload dist/gpu_pac-0.3.2*"
echo ""
echo "Note: You'll need to enter your PyPI username and password/token."
echo "For tokens, use __token__ as username and the token as password."