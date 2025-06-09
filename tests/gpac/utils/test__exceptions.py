#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:25:33 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/utils/test__exceptions.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/utils/test__exceptions.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
from gpac.utils._exceptions import (ConfigurationError, DeviceError, GPACError,
                                    MemoryError)


class TestExceptions:
    def test_gpac_error_inheritance(self):
        assert issubclass(GPACError, Exception)

        error = GPACError("test message")
        assert str(error) == "test message"

    def test_memory_error_inheritance(self):
        assert issubclass(MemoryError, GPACError)
        assert issubclass(MemoryError, Exception)

        error = MemoryError("memory issue")
        assert str(error) == "memory issue"

    def test_configuration_error_inheritance(self):
        assert issubclass(ConfigurationError, GPACError)
        assert issubclass(ConfigurationError, Exception)

        error = ConfigurationError("config issue")
        assert str(error) == "config issue"

    def test_device_error_inheritance(self):
        assert issubclass(DeviceError, GPACError)
        assert issubclass(DeviceError, Exception)

        error = DeviceError("device issue")
        assert str(error) == "device issue"

    def test_Exception_raising(self):
        with pytest.raises(GPACError):
            raise GPACError("base error")

        with pytest.raises(MemoryError):
            raise MemoryError("out of memory")

        with pytest.raises(ConfigurationError):
            raise ConfigurationError("bad config")

        with pytest.raises(DeviceError):
            raise DeviceError("cuda error")

    def test_Exception_catching_as_base(self):
        # Test that specific exceptions can be caught as GPACError
        with pytest.raises(GPACError):
            raise MemoryError("memory issue")

        with pytest.raises(GPACError):
            raise ConfigurationError("config issue")

        with pytest.raises(GPACError):
            raise DeviceError("device issue")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
