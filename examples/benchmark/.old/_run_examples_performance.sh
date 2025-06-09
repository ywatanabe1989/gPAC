#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-02 12:56:10 (ywatanabe)"
# File: ./.claude-worktree/gPAC/examples/performance/run_examples_performance.sh

THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
echo > "$LOG_PATH"

BLACK='\033[0;30m'
LIGHT_GRAY='\033[0;37m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo_info() { echo -e "${LIGHT_GRAY}$1${NC}"; }
echo_success() { echo -e "${GREEN}$1${NC}"; }
echo_warning() { echo -e "${YELLOW}$1${NC}"; }
echo_error() { echo -e "${RED}$1${NC}"; }
# ---------------------------------------

main() {
    find ./examples/performance -name "*.py" | grep -v "/\." | while read -r file_path; do
        echo "$file_path"
        chmod +x "$file_path"
        python "$file_path"
    done
}
main | tee "$LOG_PATH"
# /home/ywatanabe/proj/.claude-worktree/gPAC/examples/performance/run_examples_performance.sh
  # File "/data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/gPAC/./examples/performance/.old/comprehensive_benchmark_refined_v02.py", line 47, in <module>

# EOF