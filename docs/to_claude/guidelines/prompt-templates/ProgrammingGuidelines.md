<!-- ---
!-- Timestamp: 2025-05-21 02:57:19
!-- Author: ywatanabe
!-- File: /home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/templates/ProgrammingGuidelines.md
!-- --- -->

# Programming Guidelines

## Table of Contents
- [General Coding Principles](#general-coding-principles)
- [Language-Specific Guidelines](#language-specific-guidelines)
  - [Python Guidelines](#python-guidelines)
  - [Shell Script Guidelines](#shell-script-guidelines)
  - [Elisp Guidelines](#elisp-guidelines)
- [Debugging Guidelines](#debugging-guidelines)

## General Coding Principles

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```python
# Use excessive comments
def calc(a, b):
    # This adds a and b
    # Returns the sum
    return a + b
``` | ```python
def calculate_sum(first_value, second_value):
    return first_value + second_value
``` |
| ```python
# Use one-letter variables
for i in range(10):
    for j in data[i]:
        x = j * 2
``` | ```python
# Use descriptive names
for index in range(10):
    for item in data[index]:
        doubled_value = item * 2
``` |
| ```python
# Long string lines
error_msg = f"Failed to parse JSON response: {error}\nPrompt: {self._ai_prompt}\nGenerated command: {commands}\nResponse: {response_text}"
``` | ```python
# Split strings using f-string
error_msg = (
    f"Failed to parse JSON response: {error}\n"
    f"Prompt: {self._ai_prompt}\n"
    f"Generated command: {commands}\n"
    f"Response: {response_text}"
)
``` |

- **Code Style**
  - Use 4 spaces instead of tabs for indentation in most languages
  - Keep code simple; simplicity is the ultimate sophistication
  - Code blocks should be wrapped with triple backticks and language indicator (```python)
  - Avoid unnecessary edge cases handling; readability with shorter code is prioritized

- **Naming Conventions**
  - Never use one-letter variables (like "i") for better searchability
  - Use descriptive variable, function, and class names
  - For Elisp functions, follow naming pattern: `(defun my/CATEGORY-VERB-NOUN ...` or `(defun PACKAGENAME-CATEGORY-VERB-NOUN ...`

- **Documentation**
  - Functions should have docstrings in NumPy style with examples, parameters, and returns sections
  - Comments for sections should use singular form (e.g., "# Computes..." not "# Compute...")
  - Top-level docstrings should provide overview of functionality, inputs, outputs, and prerequisites
  - Include examples of function usage in docstrings

## Language-Specific Guidelines

### Python Guidelines

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```python
def process(x, y):
    return x + y
``` | ```python
def process_values(first_value: int, second_value: int) -> int:
    """Add two integer values.
    
    Example
    -------
    >>> process_values(1, 2)
    3
    
    Parameters
    ----------
    first_value : int
        First integer to add
    second_value : int
        Second integer to add
        
    Returns
    -------
    int
        Sum of the two values
    """
    return first_value + second_value
``` |
| ```python
# Global imports
import numpy as np
import pandas as pd
import numpy as np  # Duplicate import
``` | ```python
# MECE imports
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
``` |
| ```python
# Integer without underscore
num_samples = 100000
``` | ```python
# Integer with underscore for readability
num_samples = 100_000
``` |

1. **Code Structure**
   - Export explicit type hints for functions and classes
   - Use relative imports to reduce dependencies
   - Keep imports MECE (Mutually Exclusive and Collectively Exhaustive)
   - Format integers with underscores for readability (e.g., 100_000)
   - Display integers with commas in output (e.g., 100,000)
   - Implement reusable functions and classes
   - Fixed random seed as 42 for reproducibility
   - Use modular approaches - split functions into meaningful chunks
   - Avoid try-except blocks when possible to prevent invisible errors during debugging

2. **Statistical Analysis**
   - Report statistics with p-value, significance stars, sample size, DOF, effect size, test name, statistic, and null hypothesis
   - Use FDR correction for multiple comparisons
   - Round statistical values by factor 3 and convert to .3f format
   - Format statistical results as follows:
   ```python
   results = {
      "p_value": pval,
      "stars": mngs.stats.p2stars(pval),
      "n1": n1,
      "n2": n2,
      "dof": dof,
      "effsize": effect_size,
      "test_name": test_name_text,
      "statistic": statistic_value,
      "H0": null_hypothes_text,
   }
   ```

### Shell Script Guidelines

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```bash
# Missing help option
process_file() {
  cat $1 | grep pattern
}
``` | ```bash
# With help option and usage example
process_file() {
  # Find patterns in a file
  # Example: process_file input.txt
  if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: process_file <filename>"
    return 1
  fi
  cat "$1" | grep pattern
}
``` |
| ```bash
# Incorrect syntax
if [ "$var" = "value" ]
  echo "Match"
fi
``` | ```bash
# Correct syntax
if [ "$var" = "value" ]; then
  echo "Match"
fi
``` |

1. **Script Structure**
   - Include usage information and argument parser (with -h|--help option)
   - Implement logging functionality
   - Include one-line explanation for functions with example usage
   - Ensure proper if-fi and for-do-done syntax
   - Follow template with proper logging and argument parsing

### Elisp Guidelines

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```elisp
(defun load-file (file)
  ;; Load a file
  (interactive)
  (find-file file))
``` | ```elisp
(defun my/file-load-json (json-path)
  "Load JSON file at JSON-PATH by converting to markdown first.

Example:
  (my/file-load-json \"~/.emacs.d/data/example.json\")
  ;; => Returns markdown content from converted JSON"
  (interactive "fJSON file: ")
  (let ((md-path (concat (file-name-sans-extension json-path) ".md")))
    (when (my/file-json-to-markdown json-path)
      (my/file-load-markdown md-path))))
``` |

1. **Code Structure**
   - Ensure correct parentheses count and placement
   - Follow consistent docstring style with examples
   - Use `message` function for debugging
   - Verify functions interactively for debugging
   - Follow naming conventions for functions

## Debugging Guidelines

| ❌ DO NOT | ✅ DO |
|-----------|------|
| ```python
# No debugging output
def process_data(data):
    result = transform(data)
    return result
``` | ```python
# With debugging output
def process_data(data):
    print(f"Input data: {data}")
    result = transform(data)
    print(f"Result after transform: {result}")
    return result
``` |
| ```elisp
;; No debugging
(defun process-buffer ()
  (interactive)
  (let ((content (buffer-string)))
    (transform-content content)))
``` | ```elisp
;; With debugging
(defun process-buffer ()
  (interactive)
  (let ((content (buffer-string)))
    (message "Buffer content length: %d" (length content))
    (let ((result (transform-content content)))
      (message "Transform result: %s" result)
      result)))
``` |

1. **Language-Specific Approaches**
   - Python: Add print statements
   - Elisp: Use message function
   - Shell script: Use echo statements
   - Verify syntax correctness (parentheses in Elisp, if-fi in shell)
   - Make functions interactive for testing (in Elisp)

<!-- EOF -->