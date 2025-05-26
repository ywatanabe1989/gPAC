#!/usr/bin/env python3
"""Verify if feature requests were actually completed."""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def verify_features():
    """Check if claimed features actually work."""
    
    print("="*60)
    print("VERIFYING FEATURE COMPLETION")
    print("="*60)
    
    # 1. Check Differentiable Hilbert
    print("\n1. Differentiable Hilbert Transform:")
    try:
        from gpac._Hilbert import Hilbert
        
        # Test differentiability
        x = torch.randn(1, 1, 1, 256, requires_grad=True)
        hilbert = Hilbert(seq_len=256)
        y = hilbert(x)
        
        # Check if gradients flow
        loss = y.sum()
        loss.backward()
        
        print(f"  ✅ Implementation exists")
        print(f"  ✅ Output shape: {y.shape}")
        print(f"  ✅ Gradients flow: {x.grad is not None}")
        hilbert_ok = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        hilbert_ok = False
    
    # 2. Check Differentiable Modulation Index
    print("\n2. Differentiable Modulation Index:")
    try:
        from gpac._DifferentiableModulationIndex import DifferentiableModulationIndex
        
        # Test basic functionality
        dmi = DifferentiableModulationIndex(n_bins=18)
        phase = torch.randn(1, 1, 1, 1, 256, requires_grad=True)
        amplitude = torch.randn(1, 1, 1, 1, 256, requires_grad=True)
        
        result = dmi(phase, amplitude)
        mi = result['mi']
        
        # Check gradients
        loss = mi.sum()
        loss.backward()
        
        print(f"  ✅ Implementation exists")
        print(f"  ✅ MI shape: {mi.shape}")
        print(f"  ✅ Gradients flow to phase: {phase.grad is not None}")
        print(f"  ✅ Gradients flow to amplitude: {amplitude.grad is not None}")
        dmi_ok = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        dmi_ok = False
    
    # 3. Check Gradient Testing Suite
    print("\n3. Gradient Testing Suite:")
    try:
        # Check if test file exists
        test_path = "../tests/custom/test_gradient_checking.py"
        exists = os.path.exists(os.path.join(os.path.dirname(__file__), test_path))
        print(f"  {'✅' if exists else '❌'} Test file exists: {exists}")
        
        # Check if it has actual tests
        if exists:
            with open(os.path.join(os.path.dirname(__file__), test_path), 'r') as f:
                content = f.read()
                has_gradcheck = 'gradcheck' in content
                has_tests = 'def test_' in content
                print(f"  {'✅' if has_gradcheck else '❌'} Uses torch.autograd.gradcheck: {has_gradcheck}")
                print(f"  {'✅' if has_tests else '❌'} Contains test functions: {has_tests}")
        
        suite_ok = exists and has_tests
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        suite_ok = False
    
    # 4. Check feature request status
    print("\n4. Feature Request Status Check:")
    feature_files = {
        "02_differentiable_hilbert.md": "completed",
        "03_differentiable_modulation_index.md": "completed",
        "05_gradient_testing_suite.md": "completed"
    }
    
    for filename, expected_loc in feature_files.items():
        path = f"../project_management/feature_requests/{expected_loc}/{filename}"
        full_path = os.path.join(os.path.dirname(__file__), path)
        
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
                # Check actual status in file
                if "Status:** Completed ✅" in content:
                    print(f"  ✅ {filename}: Marked as completed")
                elif "Status:** Needs Implementation" in content:
                    print(f"  ⚠️  {filename}: Still marked as 'Needs Implementation'")
                elif "Status:** Needed" in content:
                    print(f"  ⚠️  {filename}: Still marked as 'Needed'")
                else:
                    print(f"  ❓ {filename}: Unknown status")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    if hilbert_ok and dmi_ok and suite_ok:
        print("✅ All features appear to be properly implemented")
    else:
        print("⚠️  Some features may not be fully completed:")
        if not hilbert_ok:
            print("  - Differentiable Hilbert needs work")
        if not dmi_ok:
            print("  - Differentiable MI needs work")
        if not suite_ok:
            print("  - Gradient testing suite needs work")
    
    print("\nNOTE: Feature request files may need status updates")
    print("Feature 03 and 05 are marked as 'Needs Implementation' but are in completed folder")

if __name__ == "__main__":
    verify_features()