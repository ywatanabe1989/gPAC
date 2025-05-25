#!/bin/bash
# Safe removal script that moves files to trash instead of deleting

TRASH_DIR=".trash/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TRASH_DIR"

for file in "$@"; do
    if [ -e "$file" ]; then
        echo "Moving $file to $TRASH_DIR/"
        mv "$file" "$TRASH_DIR/"
    else
        echo "File not found: $file"
    fi
done