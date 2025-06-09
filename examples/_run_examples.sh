#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 11:00:52 (ywatanabe)"
# File: ./gPAC/examples/run_examples.sh

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

usage() {
    echo "Usage: $0 [DIRECTORY] [-c|--clear-outputs] [-h|--help]"
    echo ""
    echo "Arguments:"
    echo "  DIRECTORY     Directory to search for examples (default: ./examples)"
    echo ""
    echo "Options:"
    echo "  -c, --clear-outputs  Remove existing output directories before running (default: false)"
    echo "  -h, --help           Display this help message"
    echo ""
    echo "Description:"
    echo "  Runs all Python example scripts in the specified directory"
    echo ""
    echo "Example:"
    echo "  $0"
    echo "  $0 /path/to/examples --clear-outputs"
    exit 1
}

clear_outputs() {
    local target_dir="$1"
    echo_info "Removing output directories from $target_dir..."
    find "$target_dir" -type d -name "*_out" -exec rm -rf {} + 2>/dev/null || true
    echo_success "Output directories removed"
}

run_examples() {
    local target_dir="$1"
    example_count=0
    success_count=0
    while IFS= read -r -d '' script_path; do
        ((example_count++))
        echo_info "" | tee -a "$LOG_PATH"
        echo_info "========================================" | tee -a "$LOG_PATH"
        echo_info "Running: $script_path" | tee -a "$LOG_PATH"
        echo_info "========================================" | tee -a "$LOG_PATH"
        echo_info "" | tee -a "$LOG_PATH"
        chmod +x "$script_path"
        if (python "$script_path" 2>&1 | tee -a "$LOG_PATH"); then
            echo_success "✓ $script_path completed"
            ((success_count++))
        else
            echo_error "✗ $script_path failed"
        fi
    done < <(find "$target_dir" -type f -name "*.py" ! -path "*/.*" -print0)
    echo_success "Completed: $success_count/$example_count examples"
}

main() {
    local target_dir="./examples"
    local clear_outputs_flag=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--clear-outputs)
                clear_outputs_flag=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            -*)
                echo_error "Unknown option: $1"
                usage
                ;;
            *)
                target_dir="$1"
                shift
                ;;
        esac
    done
    if [[ ! -d "$target_dir" ]]; then
        echo_error "Directory not found: $target_dir"
        exit 1
    fi
    if [[ "$clear_outputs_flag" = true ]]; then
        clear_outputs "$target_dir"
    fi
    run_examples "$target_dir"
}
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@" 2>&1 | tee "$LOG_PATH"
fi

# EOF