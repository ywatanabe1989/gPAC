# SciTeX-Paper: LLM-Friendly Scientific Paper Management System

A modern, Python-based LaTeX manuscript management system designed for scientific publications. Built to work seamlessly with AI assistants like Claude Code.

## Philosophy

**Code-with-Paper**: Keep your analysis code and manuscript together. Figures and tables are symlinked from your actual analysis outputs, ensuring reproducibility and preventing disconnection between results and text.

## Key Features

### 1. **Living Documents**
- Figures/tables are symlinked from your project's output directories
- Changes in analysis automatically reflect in the manuscript
- No manual copying or outdated figures

### 2. **Scientific Best Practices**
- Built-in templates following scientific writing guidelines
- Automated validation for:
  - Abstract structure (7 sections, 150-250 words)
  - Introduction components (8 required sections)
  - Statistical reporting completeness
  - Figure standards (axes labels, units, appropriate ranges)
  - Reference formatting

### 3. **LLM-Friendly Design**
- Clear file structure for easy navigation
- Python-based commands (no complex shell scripts)
- Modular sections for incremental updates
- Validation feedback in structured format

## Quick Start

```bash
# 1. Copy scitex to your project
cp -r /path/to/scitex-paper/* /your/project/paper/

# 2. Initialize paper structure
python scitex.py init --project "Your Project Name"

# 3. Link your analysis outputs
python scitex.py link ../analysis/figures/fig1.png --type figure
python scitex.py link ../results/table1.tex --type table

# 4. Create sections with templates
python scitex.py create-section introduction
python scitex.py create-section results

# 5. Validate your manuscript
python scitex.py validate

# 6. Compile
python scitex.py compile
```

## Directory Structure

```
your_project/
├── analysis/           # Your actual analysis code
│   ├── figures/       # Generated figures
│   └── results/       # Generated tables/data
│
├── paper/             # Manuscript directory
│   ├── scitex.yaml    # Configuration
│   ├── scitex.py      # Main script
│   │
│   ├── manuscript/    # Main document
│   │   ├── main.tex
│   │   ├── sections/  # Document sections
│   │   └── links/     # Symlinks to figures/tables
│   │
│   ├── guidelines/    # Scientific writing guidelines
│   │   └── science/   # Best practices
│   │
│   ├── templates/     # Section templates
│   └── build/         # Compiled outputs
```

## Core Commands

### Document Management
```bash
# Initialize new paper
python scitex.py init --project "Project Name" --journal "Target Journal"

# Compile document
python scitex.py compile [manuscript|revision|supplementary]
python scitex.py compile --clean --view  # Clean build and open PDF

# Create sections from templates
python scitex.py create-section [abstract|introduction|methods|results|discussion]
```

### Code-with-Paper Integration
```bash
# Link analysis outputs (creates symlinks)
python scitex.py link ../analysis/fig_performance.png --type figure
python scitex.py link ../results/table_statistics.tex --type table

# Check linked resources
python scitex.py check-links  # Verify all symlinks are valid
```

### Validation & Quality Control
```bash
# Validate entire document
python scitex.py validate

# Specific validations
python scitex.py validate-abstract
python scitex.py validate-stats
python scitex.py validate-figures

# Check references
python scitex.py check-refs
```

### Utilities
```bash
# Word count
python scitex.py wordcount [--detailed]

# Create diff between versions
python scitex.py diff v1.0 current

# Package for submission
python scitex.py package --anonymous  # For peer review
python scitex.py package --final      # Camera-ready
```

## Configuration (scitex.yaml)

```yaml
project:
  name: "Your Project"
  version: "0.1.0"
  
# Link to your analysis directories
analysis:
  figures: "../analysis/figures"
  tables: "../results/tables"
  data: "../data/processed"

# LaTeX settings
latex:
  engine: pdflatex  # or xelatex, lualatex
  bibtex: bibtex    # or biber

# Validation settings
validation:
  strict: true      # Enforce all guidelines
  stats_format: "apa"  # Statistics formatting style
```

## Working with LLMs

When working with AI assistants:

1. **Section-by-Section Editing**: 
   ```
   "Please improve the introduction following the 8-section template"
   ```

2. **Validation-Driven Revision**:
   ```
   "Run validation and fix the issues found"
   ```

3. **Figure Integration**:
   ```
   "Link all figures from analysis/figures/ that are referenced in the text"
   ```

4. **Statistical Reporting**:
   ```
   "Check that all statistical reports are complete with test names, 
    statistics, df, p-values, and effect sizes"
   ```

## Scientific Writing Templates

### Abstract Structure
1. Basic Introduction (1-2 sentences)
2. Detailed Background (2-3 sentences)  
3. General Problem (1 sentence)
4. Main Result ("Here we show..." - 1 sentence)
5. Results with Comparisons (2-3 sentences)
6. General Context (1-2 sentences)
7. Broader Perspective (2-3 sentences)

### Introduction Structure
1. Opening Statement
2. Importance of the Field
3. Existing Knowledge and Gaps
4. Limitations in Previous Works
5. Research Question or Hypothesis
6. Approach and Methods
7. Overview of Results (optional)
8. Significance and Implications

### Statistical Reporting
Always include:
- Test name (t-test, ANOVA, etc.)
- Test statistic (t, F, χ², etc.)
- Degrees of freedom
- p-value
- Effect size (Cohen's d, η², etc.)
- Sample sizes
- Correction method if multiple comparisons

## Best Practices

1. **Version Control**: Commit often, tag releases
2. **Reproducibility**: Keep analysis code that generates figures/tables
3. **Organization**: One figure/table per file
4. **Naming**: Use descriptive names (fig_accuracy_comparison.py, not fig1.py)
5. **Documentation**: Comment your figure generation code

## Extending SciTeX

Create custom validators or commands:

```python
# my_validators.py
from scitex import SciTeX

class MySciTeX(SciTeX):
    def validate_custom(self):
        # Your validation logic
        pass

# Use: python my_validators.py validate-custom
```

## FAQ

**Q: How do I update a figure?**
A: Just regenerate it in your analysis code. The symlink automatically points to the updated version.

**Q: Can I use this with Overleaf?**
A: Yes, but you'll need to copy files instead of using symlinks. Use `python scitex.py package` to create a portable version.

**Q: How do I handle multiple authors' edits?**
A: Use git branches for different authors, merge carefully, and validate after merging.

## License

MIT License - Use freely in your research projects.