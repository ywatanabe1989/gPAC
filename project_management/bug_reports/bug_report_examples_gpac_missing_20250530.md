# Bug Report: Missing ./examples/gpac Directory in Claude Worktree

## Issue Summary
The `./examples/gpac` directory exists in the main develop branch but is missing from the claude-develop worktree.

## Environment
- Date: 2025-05-30
- Branch: claude-develop (worktree)
- Location: `/home/ywatanabe/proj/.claude-worktree/gPAC`

## Steps to Reproduce
1. Create claude-develop worktree from develop branch
2. Check for `./examples/gpac` directory
3. Directory is missing despite existing in main worktree

## Investigation Results

### Root Cause
The `./examples/gpac` directory contains **untracked files** that were never committed to git:
- When claude-develop worktree was created, only tracked files were included
- Untracked files in the main worktree are not shared between worktrees

### Directory Contents (in main worktree)
```
/home/ywatanabe/proj/gPAC/examples/gpac/
├── example_BandPassFilter.py
├── example_BandPassFilter_out/
├── example_Hilbert.py
├── example_Hilbert_out/
├── example_ModulationIndex.py
├── example_Profiler.py
├── example_Profiler_out/
├── example_visualization.py
├── _Filters/
└── .old/
```

## Impact
- Cannot access or work with gpac-specific examples in Claude worktree
- Examples are effectively "hidden" from version control
- Potential loss of example code if not properly tracked

## Proposed Solutions

### Solution 1: Commit files to git (Recommended)
In the main worktree:
```bash
cd ~/proj/gPAC
git add examples/gpac/
git commit -m "Add gpac examples directory"
git push origin develop
```

Then in Claude worktree:
```bash
git pull origin develop
```

### Solution 2: Copy files temporarily
```bash
cp -r ~/proj/gPAC/examples/gpac ./examples/
```

### Solution 3: Create symlink
```bash
ln -s ~/proj/gPAC/examples/gpac ./examples/gpac
```

## Prevention
- Always commit important files before creating worktrees
- Run `git status` regularly to check for untracked files
- Consider adding `.gitignore` rules for files that should remain untracked

## Status
- [ ] Files need to be committed to git
- [ ] Claude worktree needs to be updated
- [ ] Verify all examples are properly tracked

## Related Information
- This is a common issue with git worktrees: they only share tracked files
- No data loss occurred; files still exist in main worktree
- Similar untracked directories may exist elsewhere in the project