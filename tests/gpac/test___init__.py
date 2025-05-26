import pytest
import sys
import os

# Add path for gpac imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import gpac

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/__init__.py
# --------------------------------------------------------------------------------
# from ._PAC import PAC, SyntheticPACDataset
# from ._calculate_gpac import calculate_pac
# from ._SyntheticDataGenerator import SyntheticDataGenerator
# 
# __version__ = "0.2.0"

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/__init__.py
# --------------------------------------------------------------------------------
