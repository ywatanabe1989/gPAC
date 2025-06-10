# gPAC Cleanup Plan - 2025-06-11

## Overview
This plan outlines the cleanup tasks to compact the gPAC codebase to production-ready quality.

## Identified Issues

### 1. Python Cache Files
- **959** `__pycache__` directories
- **7,607** `.pyc` and `.pyo` files
- These should be removed and added to `.gitignore`

### 2. Old/Backup Directories
- **30+** `.old` directories throughout the project
- Contains obsolete versions of files
- Nested `.old` directories (e.g., `.old/.old/.old`)

### 3. Potential File Naming Issues
- Need to check for files with development patterns:
  - `*-v01`, `*-v02`, etc.
  - `*-fix`, `*-improved`, `*-enhanced`
  - `*-old`, `*-backup`, `*-tmp`

## Cleanup Tasks

### Phase 1: Python Cache Cleanup
1. Remove all `__pycache__` directories
2. Remove all `.pyc` and `.pyo` files
3. Ensure `.gitignore` includes Python cache patterns

### Phase 2: Old Directory Cleanup
1. Review contents of `.old` directories
2. Determine if any files need to be preserved
3. Remove empty `.old` directories
4. Consolidate nested `.old` structures

### Phase 3: File Naming Standardization
1. Identify files with development suffixes
2. Rename to clean production names
3. Consolidate duplicate functionality

### Phase 4: Documentation Cleanup
1. Remove obsolete documentation
2. Update READMEs where necessary
3. Ensure all docs are current

### Phase 5: Final Verification
1. Run all tests to ensure nothing is broken
2. Verify the codebase builds correctly
3. Check that examples still work

## Safety Measures
- Created feature branch: `feature/cleanup-2025-0611-031100`
- Will use `./docs/to_claude/bin/general/safe_rm.sh` for all removals
- Will create detailed log of all changes

## Estimated Impact
- Significant reduction in repository size
- Cleaner, more professional codebase
- Easier navigation and maintenance

---
Created: 2025-06-11 03:11:00
Agent: Claude (e636b143-8653-4143-b7b4-b32f7cf0aa40)