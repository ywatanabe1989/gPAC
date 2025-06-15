#!/bin/bash
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-01 06:37:04 (ywatanabe)"
# File: ./.claude-worktree/gPAC/examples/sync.sh
=======
# Timestamp: "2025-01-29 21:00:00 (ywatanabe)"
# File: ./gPAC/examples/sync_examples_with_source.sh
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf

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
<<<<<<< HEAD
touch "$LOG_PATH" >/dev/null 2>&1
=======

touch "$LOG_PATH" >/dev/null 2>&1

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
THIS_DIR="./examples"
ORIG_DIR="$(pwd)"
ROOT_DIR="$(realpath $THIS_DIR/..)"
cmd="cd $ROOT_DIR" && echo "$cmd" && eval "$cmd"
<<<<<<< HEAD
# Set up colors for terminal output
PURPLE='\033[0;35m'
=======

# Set up colors for terminal output
PURPLE='\033[0;35m'

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
########################################
# Usage & Argument Parser
########################################
# Default Values
DO_MOVE=false
<<<<<<< HEAD
CLEAR_OUTPUT=false
SRC_DIR="$(realpath "${THIS_DIR}/../src/gpac")"
EXAMPLES_DIR="$(realpath "${THIS_DIR}/../examples/gpac")"
=======
SRC_DIR="$(realpath "${THIS_DIR}/../src/gpac")"
EXAMPLES_DIR="$(realpath "${THIS_DIR}/../examples/gpac")"

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Creates example file structure mirroring source files for module-specific examples."
    echo
    echo "Options:"
    echo "  -m, --move         Move stale example files to .old directory instead of just reporting (default: $DO_MOVE)"
<<<<<<< HEAD
    echo "  -c, --clear-output Clear all outputs directories (default: $CLEAR_OUTPUT)"
=======
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    echo "  -s, --source DIR   Specify custom source directory (default: $SRC_DIR)"
    echo "  -e, --examples DIR Specify custom examples directory (default: $EXAMPLES_DIR)"
    echo "  -h, --help         Display this help message"
    echo
    echo "Example:"
    echo "  $0 --move"
    echo "  $0 --source /path/to/src --examples /path/to/examples"
<<<<<<< HEAD
    echo "  $0 --clear-output"
    exit 1
}
=======
    exit 1
}

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--move)
            DO_MOVE=true
            shift
            ;;
<<<<<<< HEAD
        -c|--clear-output)
            CLEAR_OUTPUT=true
            shift
            ;;
=======
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
        -s|--source)
            SRC_DIR="$2"
            shift 2
            ;;
        -e|--examples)
            EXAMPLES_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
# Set default directories if not specified
if [ -z "$SRC_DIR" ]; then
    cmd="cd $ROOT_DIR" && echo "$cmd" && eval "$cmd"
fi
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
########################################
# Example Structure
########################################
prepare_examples_structure_as_source() {
    [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1
    construct_blacklist_patterns
    find "$SRC_DIR" -type d "${FIND_EXCLUDES[@]}" | while read -r dir; do
        examples_dir="${dir/$SRC_DIR/$EXAMPLES_DIR}"
        mkdir -p "$examples_dir"
    done
}
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
########################################
# Example Template
########################################
get_example_template() {
    local src_file=$1
    local module_name=$(basename "${src_file%.py}")
    local src_rel_path="${src_file#$SRC_DIR/}"
<<<<<<< HEAD
=======
    
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    cat << EOL
#!/usr/bin/env python3
"""
Example: Using ${module_name} module
<<<<<<< HEAD
This example demonstrates the usage of ${module_name} from gPAC.
Source: ${src_rel_path}
"""
=======

This example demonstrates the usage of ${module_name} from gPAC.

Source: ${src_rel_path}
"""

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
import numpy as np
import torch
import scitex as stx
from gpac import ${module_name}
<<<<<<< HEAD
=======


>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
def main():
    """Main example function."""
    # Set random seed for reproducibility
    stx.gen.fix_seeds(42)
<<<<<<< HEAD
    # Create output directory using stx
    sdir = stx.io.get_dirpath(__file__, "outputs")
=======
    
    # Create output directory using stx
    sdir = stx.io.get_dirpath(__file__, "outputs")
    
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    # TODO: Add example code for ${module_name}
    print(f"Example for ${module_name} module")
    print("This is a placeholder example file.")
    print(f"Please implement examples demonstrating {module_name} usage.")
<<<<<<< HEAD
=======
    
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    # Example structure:
    # 1. Initialize the module
    # 2. Create sample data
    # 3. Process data
    # 4. Visualize results
    # 5. Save outputs using stx
<<<<<<< HEAD
    print("\\nExample completed!")
=======
    
    print("\\nExample completed!")


>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
if __name__ == "__main__":
    main()
EOL
}
<<<<<<< HEAD
update_example_file() {
    local example_file=$1
    local src_file=$2
=======

update_example_file() {
    local example_file=$1
    local src_file=$2
    
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    if [ ! -f "$example_file" ]; then
        # If file doesn't exist, create it with template
        echo "$example_file not found. Creating..."
        mkdir -p "$(dirname "$example_file")"
<<<<<<< HEAD
=======
        
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
        # Create with example template
        get_example_template "$src_file" > "$example_file"
        chmod +x "$example_file"
        echo_success "Created: $example_file"
    else
        echo_info "Exists: $example_file (preserving existing content)"
    fi
}
<<<<<<< HEAD
########################################
# Output Cleaner
########################################
clear_output_directories() {
    if [ "$CLEAR_OUTPUT" = "true" ]; then
        echo "Clearing output directories..."
        find "$THIS_DIR" -type d -name "outputs" -not -path "*.old*" | while read -r output_dir; do
            if [ -d "$output_dir" ]; then
                echo_warning "Removing: $output_dir"
                rm -rf "$output_dir"
            fi
        done
        echo_success "Output directories cleared"
    fi
}
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
########################################
# Finder
########################################
construct_blacklist_patterns() {
    local EXCLUDE_PATHS=(
        "*/.*"
        "*/.*/*"
        "*/deprecated*"
        "*/archive*"
        "*/backup*"
        "*/tmp*"
        "*/temp*"
        "*/RUNNING/*"
        "*/FINISHED/*"
        "*/FINISHED_SUCCESS/*"
        "*/2025Y*"
        "*/2024Y*"
        "*/__pycache__/*"
        "*/__init__.py"  # Skip __init__.py files for examples
    )
<<<<<<< HEAD
    if [ "$CLEAR_OUTPUT" = "false" ]; then
        EXCLUDE_PATHS+=("*/outputs/*")
    fi
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    FIND_EXCLUDES=()
    PRUNE_ARGS=()
    for path in "${EXCLUDE_PATHS[@]}"; do
        FIND_EXCLUDES+=( -not -path "$path" )
        PRUNE_ARGS+=( -path "$path" -o )
    done
    unset 'PRUNE_ARGS[${#PRUNE_ARGS[@]}-1]'
}
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
find_files() {
    local search_path=$1
    local type=$2
    local name_pattern=$3
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    construct_blacklist_patterns
    find "$search_path" \
        \( "${PRUNE_ARGS[@]}" \) -prune -o -type "$type" -name "$name_pattern" -print
}
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
########################################
# Clean-upper
########################################
move_stale_example_files_to_old() {
    local timestamp="$(date +%Y%m%d_%H%M%S)"
<<<<<<< HEAD
    find "$EXAMPLES_DIR" -name "example_*.py" -not -path "*.old*" | while read -r example_path; do
=======

    find "$EXAMPLES_DIR" -name "example_*.py" -not -path "*.old*" | while read -r example_path; do

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
        # Determine corresponding source file
        example_rel_path="${example_path#$EXAMPLES_DIR/}"
        example_rel_dir="$(dirname $example_rel_path)"
        example_filename="$(basename $example_rel_path)"
<<<<<<< HEAD
        # Extract module name from example filename
        # Pattern: example_module_name.py -> _ModuleName.py or module_name.py
        # First remove "example_" prefix
        module_part="${example_filename#example_}"
        # Convert snake_case to CamelCase for source files with underscore prefix
        # e.g., pac_analysis -> PAC, bandpass_filter -> BandPassFilter
        camel_case_module=$(echo "$module_part" | sed -e 's/_\([a-z]\)/\U\1/g' -e 's/^./\U&/' -e 's/\.py$//')
        # Try multiple possible source file names
        src_filename_with_underscore="_${camel_case_module}.py"
        src_filename_snake_case="${module_part}"
        src_filename_simple="_${module_part}"
        src_rel_dir="$example_rel_dir"
        # Check all possible source file names
        src_path_with_underscore="$SRC_DIR/$src_rel_dir/$src_filename_with_underscore"
        src_path_snake_case="$SRC_DIR/$src_rel_dir/$src_filename_snake_case"
        src_path_simple="$SRC_DIR/$src_rel_dir/$src_filename_simple"
        # If none of the source files exist, it's a stale example
        if [ ! -f "$src_path_with_underscore" ] && [ ! -f "$src_path_snake_case" ] && [ ! -f "$src_path_simple" ] && [ -f "$example_path" ]; then
=======

        # Extract module name from example filename
        # Assuming pattern: example_ModuleName.py -> _ModuleName.py
        src_filename="${example_filename#example}"
        if [[ ! "$src_filename" =~ ^_ ]]; then
            src_filename="_${src_filename}"
        fi
        
        src_rel_dir="$example_rel_dir"
        src_rel_path="$src_rel_dir/$src_filename"
        src_path="$SRC_DIR/$src_rel_path"

        if [ ! -f "$src_path" ] && [ -f "$example_path" ]; then
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
            stale_example_path=$example_path
            stale_example_filename="$(basename $stale_example_path)"
            stale_example_path_dir="$(dirname $stale_example_path)"
            old_dir_with_timestamp="$stale_example_path_dir/.old-$timestamp"
            tgt_path="$old_dir_with_timestamp/$stale_example_filename"
<<<<<<< HEAD
            echo -e "${RED}Stale Example       : $stale_example_path${NC}"
            echo -e "${RED}If you want to remove this stale example file, please run $0 -m${NC}"
=======

            echo -e "${RED}Stale Example       : $stale_example_path${NC}"
            echo -e "${RED}If you want to remove this stale example file, please run $0 -m${NC}"

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
            if [ "$DO_MOVE" = "true" ]; then
                # Ensure target dir exists
                mkdir -p "$old_dir_with_timestamp"
                # Move file
                mv "$stale_example_path" "$tgt_path"
                echo -e "${GREEN}Moved: $stale_example_path -> $tgt_path${NC}"
            fi
        fi
    done
}
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
########################################
# Main
########################################
main() {
    local do_move=${1:-false}
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    echo "Using SRC_DIR: $SRC_DIR"
    echo "Using EXAMPLES_DIR: $EXAMPLES_DIR"
    echo ""
    echo "Note: This creates placeholder example files for each source module."
    echo "Only important modules need actual example implementations."
    echo ""
<<<<<<< HEAD
    # Clear outputs if requested
    clear_output_directories
    # Create examples/gpac directory if it doesn't exist
    mkdir -p "$EXAMPLES_DIR"
=======

    # Create examples/gpac directory if it doesn't exist
    mkdir -p "$EXAMPLES_DIR"

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    # Only create examples for key modules (not every single source file)
    local KEY_MODULES=(
        "_PAC.py"
        "_BandPassFilter.py"
        "_Hilbert.py"
        "_ModulationIndex.py"
        "_SyntheticDataGenerator.py"
        "_Profiler.py"
    )
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    # Process each key module
    for module in "${KEY_MODULES[@]}"; do
        find_files "$SRC_DIR" f "$module" | while read -r src_file; do
            # Skip if in subdirectory we don't want examples for
            [[ "$src_file" =~ /PackageHandlers/ ]] && continue
<<<<<<< HEAD
=======
            
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
            # derive relative path and parts
            rel="${src_file#$SRC_DIR/}"
            rel_dir=$(dirname "$rel")
            src_base=$(basename "$rel")
<<<<<<< HEAD
            # ensure example subdir exists
            examples_dir="$EXAMPLES_DIR/$rel_dir"
            mkdir -p "$examples_dir"
=======

            # ensure example subdir exists
            examples_dir="$EXAMPLES_DIR/$rel_dir"
            mkdir -p "$examples_dir"

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
            # build correct example file path
            # Convert _ModuleName.py to example_ModuleName.py
            example_base="example${src_base}"
            example_file="$examples_dir/$example_base"
<<<<<<< HEAD
=======

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
            # Process each file
            update_example_file "$example_file" "$src_file"
        done
    done
<<<<<<< HEAD
    # Also create general examples in the root examples directory
    echo ""
    echo "Creating/checking general examples..."
    # These are already created, just check they exist
    local GENERAL_EXAMPLES=(
        "example_pac_analysis.py"
        "example_bandpass_filter.py"
        "example_profiler.py"
    )
=======

    # Also create general examples in the root examples directory
    echo ""
    echo "Creating/checking general examples..."
    
    # These are already created, just check they exist
    local GENERAL_EXAMPLES=(
        "example_pac_analysis.py"
        "example_bandpass_filter.py" 
        "example_profiler.py"
    )
    
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
    for example in "${GENERAL_EXAMPLES[@]}"; do
        if [ -f "${THIS_DIR}/$example" ]; then
            echo_success "Found: ${THIS_DIR}/$example"
        else
            echo_warning "Missing: ${THIS_DIR}/$example"
        fi
    done
<<<<<<< HEAD
    # Clean up stale files
    move_stale_example_files_to_old
    # Show structure
    echo ""
    echo "Examples structure:"
    tree_exclude="outputs|__pycache__|*.pyc|.old*"
    if [ "$CLEAR_OUTPUT" = "false" ]; then
        tree_exclude="__pycache__|*.pyc|.old*"
    fi
    tree "$THIS_DIR" -I "$tree_exclude" 2>&1 | tee -a "$LOG_PATH"
}
=======

    # Clean up stale files
    move_stale_example_files_to_old

    # Show structure
    echo ""
    echo "Examples structure:"
    tree "$THIS_DIR" -I "outputs|__pycache__|*.pyc|.old*" 2>&1 | tee -a "$LOG_PATH"
}

>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
main "$@"
cd $ORIG_DIR

# EOF