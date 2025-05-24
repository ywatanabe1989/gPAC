<!-- ---
!-- Timestamp: 2025-05-21 03:34:19
!-- Author: ywatanabe
!-- File: /home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/templates/ProgrammingTest.md
!-- --- -->

# Test Code Writing Guidelines

## Table of Contents
- [Your Role](#your-role)
- [Request Overview](#request-overview)
- [General Testing Principles](#general-testing-principles)
- [Language-Specific Testing Guidelines](#language-specific-testing-guidelines)
  - [Python Testing Guidelines](#python-testing-guidelines)
  - [Elisp Testing Guidelines](#elisp-testing-guidelines)
  - [Shell Script Testing Guidelines](#shell-script-testing-guidelines)
- [Test Structure Examples](#test-structure-examples)

## Your Role
You are an experienced programmer, specializing in writing test code. Please implement, revise, debug, or refactor test code.

## Request Overview
- Determine which language is being used (Python, Elisp, shell script, etc.)
- Provide test code following the appropriate testing framework and guidelines
- Add debugging lines if similar requests are made multiple times
- Request additional source code if needed

## General Testing Principles

| ❌ DO NOT | ✅ DO |
|-----------|------|
| Include unnecessary comments | Keep code self-explanatory with meaningful names |
| Use one-letter variables (e.g., `x`) | Use descriptive variable names (e.g., `iteration_count`) |
| Create large test functions | Keep each test function small and focused |
| Test multiple behaviors in one test | Test one specific behavior per test |
| Modify source code for testing | Create proper test fixtures and mocks |
| Write ambiguous test names | Name tests clearly (e.g., `test_user_login_with_valid_credentials`) |

## Language-Specific Testing Guidelines

### Python Testing Guidelines

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```python
import unittest

class TestCalculator(unittest.TestCase):
    def test_multiple_functions(self):
        # Tests add, subtract, and multiply in one test
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(multiply(2, 3), 6)
``` | ```python
import pytest

def test_add_returns_sum_of_two_numbers():
    # Tests only one function with clear purpose
    assert add(2, 3) == 5

def test_subtract_returns_difference_of_two_numbers():
    assert subtract(5, 3) == 2

def test_multiply_returns_product_of_two_numbers():
    assert multiply(2, 3) == 6
``` |
| ```python
def test_function_without_type_hints(input_val):
    result = process(input_val)
    assert result > 0
``` | ```python
def test_process_returns_positive_value(input_val: int) -> None:
    """Test that process returns positive values.
    
    Parameters
    ----------
    input_val : int
        Test input value
    """
    result = process(input_val)
    assert result > 0
``` |

#### Python-Specific Rules
- Use pytest, not unittest
- Prepare pytest.ini configuration file
- Follow this structure:
  - ./src/project-name/__init__.py
  - ./tests/project-name/...
- Each test function should test one thing
- Each test file should contain one test class/function
- Each test class should be defined in a dedicated script
- Implement run_tests.sh (or run_tests.ps1 for Windows) in the project root
- Fix random seed as 42 for reproducibility

### Elisp Testing Guidelines

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```elisp
(ert-deftest test-multiple-functions ()
  (should (= (add 2 3) 5))
  (should (= (subtract 5 3) 2))
  (should (= (multiply 2 3) 6)))
``` | ```elisp
(ert-deftest test-add-returns-sum-of-two-numbers ()
  (should (= (add 2 3) 5)))

(ert-deftest test-subtract-returns-difference-of-two-numbers ()
  (should (= (subtract 5 3) 2)))

(ert-deftest test-multiply-returns-product-of-two-numbers ()
  (should (= (multiply 2 3) 6)))
``` |
| ```elisp
;; Modifying global state for test
(setq test-variable "test-value")
(ert-deftest test-with-modified-state ()
  (should (string= test-variable "test-value")))
``` | ```elisp
;; Preserving state during test
(ert-deftest test-with-preserved-state ()
  (let ((original-value (symbol-value 'variable-to-test)))
    (unwind-protect
        (progn
          ;; Test logic here
          (should t))
      ;; Cleanup - restore original state
      (set 'variable-to-test original-value))))
``` |

#### Elisp-Specific Rules
- Test code runs in the runtime environment
- Do not change variables for testing purposes
- Do not use SETQ/DEFVAR/DEFCUSTOM in tests
- Do not use LET/LET* for test variables
- Do not LOAD anything in tests
- Create test files with test- prefix matching source files
- Each ert-deftest should include only one should or should-not
- Preserve the environment state before and after tests

### Shell Script Testing Guidelines

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```bash
# No setup or teardown
test_function() {
  mkdir temp_dir
  touch temp_dir/test_file
  # Test code here
  # No cleanup
}
``` | ```bash
test_function() {
  # Set up test environment
  local temp_dir=$(mktemp -d)
  touch "$temp_dir/test_file"
  
  # Test code here
  
  # Clean up
  rm -rf "$temp_dir"
}
``` |
| ```bash
# Hardcoded paths
test_function() {
  ./my_script.sh /home/user/data
  # Test assertions
}
``` | ```bash
test_function() {
  local test_data_dir=$(mktemp -d)
  # Prepare test data
  
  # Run with test data
  ./my_script.sh "$test_data_dir"
  
  # Test assertions
  
  # Clean up
  rm -rf "$test_data_dir"
}
``` |

## Test Structure Examples

### Python Test Structure

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-21 03:34:19 (ywatanabe)"

"""
1. Functionality:
   - Tests the calculation module functionality
2. Input:
   - Test parameters for calculation functions
3. Output:
   - Test results indicating pass/fail status
4. Prerequisites:
   - pytest
   - calculation module
"""

import pytest
from typing import List, Dict, Any, Union
import numpy as np
from calculation import add, subtract, multiply

def test_add_returns_sum_of_two_integers() -> None:
    """Test that add function correctly sums two integers."""
    # Arrange
    first_value = 2
    second_value = 3
    expected_sum = 5
    
    # Act
    result = add(first_value, second_value)
    
    # Assert
    assert result == expected_sum
    
def test_add_handles_negative_numbers() -> None:
    """Test that add function correctly handles negative numbers."""
    # Arrange
    first_value = -2
    second_value = 3
    expected_sum = 1
    
    # Act
    result = add(first_value, second_value)
    
    # Assert
    assert result == expected_sum
```

### Elisp Test Structure

```elisp
;;; test-calculator.el --- Tests for calculator functions

;;; Commentary:
;; Tests for basic calculator functionality

;;; Code:
(require 'ert)
(require 'calculator)

(ert-deftest test-calculator-add-returns-sum-of-two-numbers ()
  "Test that `calculator-add' returns the sum of two numbers."
  (should (= (calculator-add 2 3) 5)))

(ert-deftest test-calculator-add-handles-negative-numbers ()
  "Test that `calculator-add' correctly handles negative numbers."
  (should (= (calculator-add -2 3) 1)))

(ert-deftest test-calculator-subtract-returns-difference-of-two-numbers ()
  "Test that `calculator-subtract' returns the difference of two numbers."
  (should (= (calculator-subtract 5 3) 2)))

(provide 'test-calculator)
;;; test-calculator.el ends here
```

### Shell Script Test Structure

```bash
#!/bin/bash
# test-script-name.sh
# Author: ywatanabe (ywatanabe@alumni.u-tokyo.ac.jp)
# Date: 2025-05-21-03-34

LOG_FILE="./test-results.log"

# Test setup function
setup() {
  # Create test environment
  TEST_DIR=$(mktemp -d)
  TEST_FILE="$TEST_DIR/test_data.txt"
  echo "Sample data" > "$TEST_FILE"
}

# Test teardown function
teardown() {
  # Clean up test environment
  rm -rf "$TEST_DIR"
}

# Test function
test_process_file_returns_success_for_valid_file() {
  # Test that process_file returns 0 for valid file
  ../scripts/process_file.sh "$TEST_FILE"
  local result=$?
  
  # Assert result
  if [ $result -eq 0 ]; then
    echo "PASS: process_file returns success for valid file"
    return 0
  else
    echo "FAIL: process_file returned $result instead of 0"
    return 1
  fi
}

# Main test runner
main() {
  local passed=0
  local failed=0
  
  setup
  
  # Run tests
  if test_process_file_returns_success_for_valid_file; then
    passed=$((passed+1))
  else
    failed=$((failed+1))
  fi
  
  # Add more test calls here
  
  teardown
  
  # Report results
  echo "Test Results: $passed passed, $failed failed"
  
  # Return non-zero if any tests failed
  [ $failed -eq 0 ]
}

# Run tests and log output
main "$@" 2>&1 | tee "$LOG_FILE"
```

----------
Now, my input is as follows:
----------
PLACEHOLDER

<!-- EOF -->