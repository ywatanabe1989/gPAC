#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:02:44 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__configs.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/dataset/test__configs.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
from gpac.dataset._configs import (multi_class_multi_pac_config,
                                   multi_class_single_pac_config,
                                   single_class_multi_pac_config,
                                   single_class_single_pac_config)

class TestDatasetConfigs:
    """Test predefined dataset configurations."""

    def test_single_class_single_pac_config(self):
        """Test single class single PAC configuration."""
        config = single_class_single_pac_config
        assert "single_pac" in config
        assert "components" in config["single_pac"]
        assert "noise_levels" in config["single_pac"]

        components = config["single_pac"]["components"]
        assert len(components) == 1
        assert "pha_hz" in components[0]
        assert "amp_hz" in components[0]
        assert "strength" in components[0]
        assert components[0]["pha_hz"] == 8.0
        assert components[0]["amp_hz"] == 80.0
        assert components[0]["strength"] == 0.5

        noise_levels = config["single_pac"]["noise_levels"]
        assert len(noise_levels) == 3
        assert all(isinstance(nn, float) for nn in noise_levels)

    def test_single_class_multi_pac_config(self):
        """Test single class multi PAC configuration."""
        config = single_class_multi_pac_config
        assert "multi_pac" in config
        assert "components" in config["multi_pac"]
        assert "noise_levels" in config["multi_pac"]

        components = config["multi_pac"]["components"]
        assert len(components) == 3

        for comp in components:
            assert "pha_hz" in comp
            assert "amp_hz" in comp
            assert "strength" in comp
            assert isinstance(comp["pha_hz"], float)
            assert isinstance(comp["amp_hz"], float)
            assert isinstance(comp["strength"], float)

    def test_multi_class_single_pac_config(self):
        """Test multi-class single PAC configuration."""
        config = multi_class_single_pac_config
        assert "no_pac" in config
        assert "theta_gamma" in config
        assert "alpha_beta" in config

        assert config["no_pac"]["components"] == []
        assert "noise_levels" in config["no_pac"]

        theta_gamma = config["theta_gamma"]
        assert len(theta_gamma["components"]) == 1
        assert theta_gamma["components"][0]["pha_hz"] == 8.0
        assert theta_gamma["components"][0]["amp_hz"] == 80.0

        alpha_beta = config["alpha_beta"]
        assert len(alpha_beta["components"]) == 1
        assert alpha_beta["components"][0]["pha_hz"] == 10.0
        assert alpha_beta["components"][0]["amp_hz"] == 20.0

    def test_multi_class_multi_pac_config(self):
        """Test multi-class multi PAC configuration."""
        config = multi_class_multi_pac_config
        assert "no_pac" in config
        assert "single_pac" in config
        assert "dual_pac" in config

        assert len(config["no_pac"]["components"]) == 0
        assert len(config["single_pac"]["components"]) == 1
        assert len(config["dual_pac"]["components"]) == 2

    def test_config_consistency(self):
        """Test consistency across all configurations."""
        configs = [
            single_class_single_pac_config,
            single_class_multi_pac_config,
            multi_class_single_pac_config,
            multi_class_multi_pac_config,
        ]
        for config in configs:
            assert isinstance(config, dict)
            for class_name, class_config in config.items():
                assert "components" in class_config
                assert "noise_levels" in class_config
                assert isinstance(class_config["components"], list)
                assert isinstance(class_config["noise_levels"], list)

                for comp in class_config["components"]:
                    assert "pha_hz" in comp
                    assert "amp_hz" in comp
                    assert "strength" in comp

    def test_noise_levels_consistency(self):
        """Test that all configs use consistent noise levels."""
        configs = [
            single_class_single_pac_config,
            single_class_multi_pac_config,
            multi_class_single_pac_config,
            multi_class_multi_pac_config,
        ]

        all_noise_levels = []
        for config in configs:
            for class_config in config.values():
                all_noise_levels.append(class_config["noise_levels"])

        first_noise = all_noise_levels[0]
        for noise_levels in all_noise_levels:
            assert noise_levels == first_noise

    def test_frequency_ranges(self):
        """Test that frequency values are in reasonable ranges."""
        configs = [
            single_class_single_pac_config,
            single_class_multi_pac_config,
            multi_class_single_pac_config,
            multi_class_multi_pac_config,
        ]
        for config in configs:
            for class_config in config.values():
                for comp in class_config["components"]:
                    assert 0 < comp["pha_hz"] <= 30
                    assert comp["amp_hz"] > comp["pha_hz"]
                    assert 0 < comp["strength"] <= 1

    def test_config_imports(self):
        """Test that configs can be imported from the dataset module."""
        from gpac import dataset

        assert hasattr(dataset, "single_class_single_pac_config")
        assert hasattr(dataset, "single_class_multi_pac_config")
        assert hasattr(dataset, "multi_class_single_pac_config")
        assert hasattr(dataset, "multi_class_multi_pac_config")


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
