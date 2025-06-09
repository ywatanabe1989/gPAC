#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:30:03 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/memory/test__get_device_info.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/memory/test__get_device_info.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest

from unittest.mock import patch

from gpac.memory._get_device_info import get_device_info


class TestGetDeviceInfo:
    @patch("torch.cuda.is_available", return_value=False)
    def test_no_cuda_available(self, mock_cuda):
        info = get_device_info()
        assert info == {"cuda_available": False}

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.get_device_name", return_value="RTX 4090")
    @patch("torch.cuda.get_device_properties")
    def test_cuda_available(
        self, mock_props, mock_name, mock_current, mock_count, mock_cuda
    ):
        mock_props.return_value.total_memory = 24 * 1e9  # 24GB

        info = get_device_info()

        assert info["cuda_available"] is True
        assert info["device_count"] == 2
        assert info["current_device"] == 0
        assert info["device_name"] == "RTX 4090"
        assert info["memory_gb"] == 24.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.get_device_name", return_value="Tesla V100")
    @patch("torch.cuda.get_device_properties")
    def test_cuda_available_single_device(
        self, mock_props, mock_name, mock_current, mock_count, mock_cuda
    ):
        mock_props.return_value.total_memory = 32 * 1e9  # 32GB

        info = get_device_info()

        assert info["cuda_available"] is True
        assert info["device_count"] == 1
        assert info["current_device"] == 0
        assert info["device_name"] == "Tesla V100"
        assert info["memory_gb"] == 32.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=4)
    @patch("torch.cuda.current_device", return_value=2)
    @patch("torch.cuda.get_device_name", return_value="A100-SXM4-80GB")
    @patch("torch.cuda.get_device_properties")
    def test_cuda_multi_device(
        self, mock_props, mock_name, mock_current, mock_count, mock_cuda
    ):
        mock_props.return_value.total_memory = 80 * 1e9  # 80GB

        info = get_device_info()

        assert info["cuda_available"] is True
        assert info["device_count"] == 4
        assert info["current_device"] == 2
        assert info["device_name"] == "A100-SXM4-80GB"
        assert info["memory_gb"] == 80.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
