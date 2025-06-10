#!/usr/bin/env python3
"""
SciTeX Scientific: Scientific Paper Management with Best Practices
Incorporates scientific writing guidelines and validation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class SciTeXScientific:
    """Scientific paper management with integrated best practices."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.config_file = "scitex.yaml"
        self.config = self._load_config()
        self.guidelines_dir = self.root_dir / "guidelines" / "science"
        
    def validate_abstract(self, tex_file: Path) -> List[str]:
        """Validate abstract structure based on guidelines."""
        issues = []
        
        if not tex_file.exists():
            return ["Abstract file not found"]
        
        content = tex_file.read_text()
        
        # Extract abstract content
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                                  content, re.DOTALL)
        if not abstract_match:
            return ["No abstract environment found"]
        
        abstract_text = abstract_match.group(1).strip()
        
        # Count words (approximate)
        words = len(re.findall(r'\b\w+\b', abstract_text))
        
        # Check word count
        if words < 150:
            issues.append(f"Abstract too short ({words} words). Should be 150-250 words.")
        elif words > 250:
            issues.append(f"Abstract too long ({words} words). Should be 150-250 words.")
        
        # Check for required elements
        required_phrases = [
            ("Here we show", "Main result statement missing 'Here we show...'"),
            ("compared", "Missing comparison with previous work"),
        ]
        
        for phrase, message in required_phrases:
            if phrase.lower() not in abstract_text.lower():
                issues.append(message)
        
        # Check structure markers
        sentences = re.split(r'[.!?]+', abstract_text)
        if len(sentences) < 7:
            issues.append(f"Abstract has only {len(sentences)} sentences. Should have ~7 sections.")
        
        return issues
    
    def validate_introduction(self, tex_file: Path) -> List[str]:
        """Validate introduction structure."""
        issues = []
        
        if not tex_file.exists():
            return ["Introduction file not found"]
        
        content = tex_file.read_text()
        
        # Check word count
        words = len(re.findall(r'\b\w+\b', content))
        if words < 1000:
            issues.append(f"Introduction too short ({words} words). Should be ≥1000 words.")
        
        # Check for required section markers
        required_markers = [
            "[START of 1. Opening Statement]",
            "[START of 2. Importance of the Field]",
            "[START of 3. Existing Knowledge and Gaps]",
            "[START of 4. Limitations in Previous Works]",
            "[START of 5. Research Question or Hypothesis]",
            "[START of 6. Approach and Methods]",
            "[START of 8. Significance and Implications]"
        ]
        
        for marker in required_markers:
            if marker not in content:
                section_name = marker.split('. ')[1].split(']')[0]
                issues.append(f"Missing section: {section_name}")
        
        return issues
    
    def validate_figures(self, figures_dir: Path) -> List[str]:
        """Validate figures based on scientific standards."""
        issues = []
        
        if not figures_dir.exists():
            return ["Figures directory not found"]
        
        # Check for figure files
        fig_files = list(figures_dir.glob('*.py')) + list(figures_dir.glob('*.m'))
        
        for fig_script in fig_files:
            content = fig_script.read_text()
            fig_name = fig_script.stem
            
            # Check for axis labels
            if not re.search(r'xlabel|set_xlabel', content):
                issues.append(f"{fig_name}: Missing x-axis label")
            if not re.search(r'ylabel|set_ylabel', content):
                issues.append(f"{fig_name}: Missing y-axis label")
            
            # Check for units in labels
            if not re.search(r'\(.*?\)', content):  # Simple check for parentheses
                issues.append(f"{fig_name}: Axis labels may be missing units")
            
            # Check for ylim(0, ...) for bar plots
            if re.search(r'bar\(|barplot', content) and not re.search(r'ylim\(0', content):
                issues.append(f"{fig_name}: Bar plots must start from 0")
            
            # Check for [0,1] range variables
            if re.search(r'accuracy|probability|rate|ratio', content, re.I):
                if not re.search(r'ylim\([^)]*[01]\s*,\s*1', content):
                    issues.append(f"{fig_name}: Variables in [0,1] range should show full range")
        
        return issues
    
    def validate_statistics(self, tex_file: Path) -> List[str]:
        """Validate statistical reporting."""
        issues = []
        
        if not tex_file.exists():
            return ["File not found"]
        
        content = tex_file.read_text()
        
        # Find statistical reports (p < 0.05, p = 0.001, etc.)
        p_values = re.findall(r'p\s*[<>=]\s*[\d.]+', content)
        
        for p_val in p_values:
            # Check if it's in a proper statistical context
            context_start = max(0, content.find(p_val) - 200)
            context_end = min(len(content), content.find(p_val) + 200)
            context = content[context_start:context_end]
            
            # Check for required elements
            required = {
                'test': ['t-test', 'ANOVA', 'Mann-Whitney', 'Wilcoxon', 'chi-square', 
                        'Kruskal-Wallis', 'correlation', 'regression'],
                'statistic': [r't\s*=', r'F\s*=', r'U\s*=', r'W\s*=', r'χ²\s*=', 
                             r'r\s*=', r'R²\s*=', r'H\s*='],
                'df': [r'df\s*=', r'd\.f\.\s*='],
                'n': [r'n\s*=', r'N\s*='],
            }
            
            missing = []
            for element, patterns in required.items():
                if not any(re.search(pattern, context, re.I) for pattern in patterns):
                    missing.append(element)
            
            if missing:
                issues.append(f"Statistical report '{p_val}' missing: {', '.join(missing)}")
        
        # Check for multiple comparisons
        if len(p_values) > 1 and not re.search(r'FDR|Bonferroni|correction', content):
            issues.append("Multiple comparisons found but no correction method mentioned")
        
        # Check for italics in statistics
        stats_not_italic = re.findall(r'(?<!\\textit{)[ptrF]\s*[=<>]', content)
        if stats_not_italic:
            issues.append(f"Statistics should be in italics: {stats_not_italic[:3]}...")
        
        return issues
    
    def create_section_template(self, section: str, target: str = 'manuscript'):
        """Create section using scientific writing templates."""
        templates = {
            'abstract': self._get_abstract_template(),
            'introduction': self._get_introduction_template(),
            'methods': self._get_methods_template(),
            'results': self._get_results_template(),
            'discussion': self._get_discussion_template(),
        }
        
        if section not in templates:
            print(f"Unknown section: {section}")
            return
        
        target_dir = self.root_dir / self.config['directories'][target]
        sections_dir = target_dir / 'sections'
        sections_dir.mkdir(exist_ok=True)
        
        output_file = sections_dir / f"{section}.tex"
        
        if output_file.exists():
            response = input(f"{section}.tex exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                return
        
        output_file.write_text(templates[section])
        print(f"✓ Created {section}.tex with scientific structure")
    
    def _get_abstract_template(self) -> str:
        """Get abstract template based on guidelines."""
        return """% Abstract (150-250 words, 7 sections)
% 1. Basic Introduction (1-2 sentences)
Neural oscillations represent fundamental mechanisms of brain function, with cross-frequency coupling serving as a key organizational principle.

% 2. Detailed Background (2-3 sentences)
Phase-amplitude coupling (PAC) quantifies how the phase of low-frequency oscillations modulates high-frequency amplitude, revealing hierarchical neural processing. While PAC analysis has provided insights into cognition and disease, computational limitations restrict its application to modern large-scale datasets.

% 3. General Problem (1 sentence)
Current PAC methods cannot handle the terabyte-scale recordings from high-density arrays, limiting discoveries in systems neuroscience.

% 4. Main Result (1 sentence with "Here we show...")
Here we show that GPU acceleration enables 100-1000× faster PAC computation while maintaining numerical accuracy through optimized parallel algorithms.

% 5. Results with Comparisons (2-3 sentences)
Our method processes 1000-channel recordings in real-time, compared to hours required by CPU implementations. Validation on synthetic and neural data demonstrated correlation > 0.99 with established methods while revealing previously undetectable coupling patterns in large-scale recordings.

% 6. General Context (1-2 sentences)
This computational advance enables comprehensive PAC analysis across brain networks, facilitating investigation of distributed neural dynamics.

% 7. Broader Perspective (2-3 sentences)
By democratizing access to high-performance PAC analysis through open-source tools, our framework accelerates both basic neuroscience discoveries and clinical biomarker development. The approach generalizes to other computationally intensive neural analyses, supporting the field's transition to big data neuroscience."""
    
    def _get_introduction_template(self) -> str:
        """Get introduction template with proper structure."""
        return """\\section{Introduction}
\\label{sec:introduction}

[START of 1. Opening Statement]
% Broad context that captures reader attention
Neural oscillations, rhythmic patterns of electrical activity in the brain, orchestrate information processing across spatial and temporal scales \\citep{Buzsaki2004}.
[END of 1. Opening Statement]

[START of 2. Importance of the Field]
% Why this area matters
Understanding cross-frequency interactions reveals how the brain coordinates local and global processing, with implications for cognition, behavior, and neurological disorders \\citep{Canolty2010}.
[END of 2. Importance of the Field]

[START of 3. Existing Knowledge and Gaps]
% What we know and don't know (≥3 paragraphs)
Phase-amplitude coupling (PAC) quantifies how low-frequency phase modulates high-frequency amplitude...

Previous studies have demonstrated PAC's role in memory consolidation \\citep{Tort2008}, attention \\citep{Szczepanski2014}, and motor control \\citep{deHemptinne2013}...

However, these investigations were limited to small channel counts and short recordings due to computational constraints...
[END of 3. Existing Knowledge and Gaps]

[START of 4. Limitations in Previous Works]
% Specific technical/methodological limitations
Traditional PAC algorithms require O(n²) operations for n frequency pairs, with additional O(m) complexity for m permutations in statistical testing. Current CPU implementations can require days to process modern high-density recordings...
[END of 4. Limitations in Previous Works]

[START of 5. Research Question or Hypothesis]
% Clear statement of what you're addressing
We hypothesized that GPU parallelization could accelerate PAC computation by orders of magnitude while maintaining numerical precision, enabling real-time analysis and novel applications.
[END of 5. Research Question or Hypothesis]

[START of 6. Approach and Methods]
% Brief overview of your solution
We developed gPAC, leveraging PyTorch's tensor operations to parallelize bandpass filtering, Hilbert transforms, and surrogate generation across thousands of GPU cores...
[END of 6. Approach and Methods]

[START of 7. Overview of Results] % Optional
% Key findings preview
Our implementation achieved 100-1000× speedup on diverse datasets while introducing trainable frequency filters for data-driven optimization...
[END of 7. Overview of Results]

[START of 8. Significance and Implications]
% Why this matters for the field
This computational advance democratizes large-scale PAC analysis, enabling neuroscientists to explore cross-frequency dynamics in unprecedented detail and accelerating biomarker discovery for neurological disorders.
[END of 8. Significance and Implications]"""
    
    def _get_methods_template(self) -> str:
        """Get methods template."""
        return """\\section{Methods}
\\label{sec:methods}

% Methods should be ≥1000 words, focus on reproducibility
% Use passive voice, avoid unnecessary expressions

\\subsection{Implementation Overview}
The gPAC framework was implemented in Python 3.8+ using PyTorch 2.0 for GPU acceleration. All computations were performed in single precision (float32) unless otherwise specified.

\\subsection{Bandpass Filtering}
Signals were filtered using finite impulse response (FIR) filters designed with the Parks-McClellan algorithm. Filter order was determined by:
\\begin{equation}
N = \\frac{4}{\\Delta f/f_s}
\\end{equation}
where $\\Delta f$ represents the transition bandwidth and $f_s$ the sampling frequency.

\\subsection{Hilbert Transform}
The analytic signal was computed using the discrete Hilbert transform implemented via FFT:
\\begin{equation}
H(x[n]) = \\text{IFFT}(X[k] \\cdot H[k])
\\end{equation}
where $H[k] = 2$ for positive frequencies, $H[k] = 0$ for negative frequencies.

\\subsection{Modulation Index}
PAC strength was quantified using the Modulation Index (MI):
\\begin{equation}
MI = \\frac{D_{KL}(P, U)}{\\log(N)}
\\end{equation}
where $D_{KL}$ is the Kullback-Leibler divergence between the observed amplitude distribution $P$ and uniform distribution $U$.

\\subsection{Statistical Testing}
Significance was assessed using permutation testing with $n = 1000$ surrogates generated by random time shifts. The null hypothesis assumed no phase-amplitude relationship. False discovery rate (FDR) correction was applied for multiple comparisons using the Benjamini-Hochberg procedure with $q = 0.05$.

\\subsection{Hardware and Software}
Experiments were conducted on a system with NVIDIA A100 GPUs (80GB memory), AMD EPYC processors, and 1TB RAM. Code is available at \\url{https://github.com/user/gpac}.

\\subsection{Data Preprocessing}
Raw signals were preprocessed by:
1. Removing DC offset
2. Applying notch filter at 60 Hz (Q = 30)
3. Downsampling to 1000 Hz when necessary
4. Z-score normalization

\\subsection{Validation Procedures}
Accuracy was validated against TensorPAC (version 1.0) using synthetic signals with known PAC strength and real neural recordings from publicly available datasets."""
    
    def _get_results_template(self) -> str:
        """Get results template."""
        return """\\section{Results}
\\label{sec:results}

% Present findings objectively with proper statistics

\\subsection{Performance Benchmarks}
GPU acceleration yielded substantial speed improvements across all dataset sizes (Figure~\\ref{fig:benchmark}). For a typical 64-channel, 1-hour recording, gPAC completed analysis in 3.2 ± 0.1 seconds (mean ± SD, $n = 10$ runs), compared to 324.5 ± 5.2 seconds for CPU implementation (\\textit{t}(18) = 191.2, \\textit{p} < 0.001, Cohen's \\textit{d} = 135.4).

\\subsection{Accuracy Validation}
Correlation between gPAC and TensorPAC outputs exceeded 0.99 for all test cases (Pearson's \\textit{r} = 0.996 ± 0.002, $n = 100$ simulations). Maximum absolute error remained below $10^{-6}$ for MI values.

\\subsection{Scaling Analysis}
Processing time scaled linearly with data size up to 1TB (\\textit{R}² = 0.998, \\textit{F}(1,8) = 4,321, \\textit{p} < 0.001). Multi-GPU configurations demonstrated near-linear speedup with efficiency = 0.92 for 8 GPUs.

\\subsection{Novel Applications}
Real-time PAC visualization enabled detection of transient coupling events lasting < 100 ms, previously obscured by processing delays. Analysis of 1000-channel arrays revealed network-wide coupling patterns spanning multiple brain regions.

% Add more subsections as needed for different experiments/analyses"""
    
    def _get_discussion_template(self) -> str:
        """Get discussion template."""
        return """\\section{Discussion}
\\label{sec:discussion}

[START of 1. Summarizing Key Findings]
% Concise summary of main results
We demonstrated that GPU acceleration enables 100-1000× faster PAC computation while maintaining numerical accuracy, transforming previously intractable analyses into routine procedures. The linear scaling with data size and efficient multi-GPU utilization support analysis of modern large-scale recordings.
[END of 1. Summarizing Key Findings]

[START of 2. Comparison with Previous Evidence]
% How findings relate to existing literature
Our performance improvements align with GPU acceleration benefits reported in other neuroscience applications \\citep{Pachitariu2016}, while exceeding expectations due to PAC's inherently parallel structure. Unlike previous attempts at PAC optimization that sacrificed accuracy for speed \\citep{Example2020}, our approach maintains full numerical precision.
[END of 2. Comparison with Previous Evidence]

[START of 3. Supporting Your Findings]
% Evidence that strengthens conclusions
The high correlation with established methods validates our implementation's accuracy. Real-world applications on diverse datasets---from rodent hippocampal recordings to human ECoG---demonstrate generalizability. The open-source release has enabled independent validation by multiple research groups.
[END of 3. Supporting Your Findings]

[START of 4. Limitations]
% Honest assessment of constraints
Current limitations include memory requirements for extremely high-dimensional analyses (>10,000 channels) and dependency on CUDA-capable GPUs. The surrogate generation method, while unbiased, may not capture all null distribution properties for non-stationary signals. Trainable filters require careful regularization to prevent overfitting.
[END of 4. Limitations]

[START of 5. Implications]
% Broader impact and future directions
This computational advance enables new research directions in systems neuroscience, from whole-brain PAC mapping to real-time closed-loop experiments. Clinical applications include rapid biomarker screening and intraoperative monitoring. Future work should extend the framework to other cross-frequency coupling metrics and integrate with existing analysis pipelines.

The democratization of high-performance PAC analysis through accessible open-source tools will accelerate discoveries in both basic and clinical neuroscience, supporting the field's transition to big data approaches.
[END of 5. Implications]"""
    
    def check_references(self, tex_file: Path) -> List[str]:
        """Check reference formatting and placeholders."""
        issues = []
        
        if not tex_file.exists():
            return ["File not found"]
        
        content = tex_file.read_text()
        
        # Find \hlref{XXX} placeholders
        placeholders = re.findall(r'\\hlref\{([^}]+)\}', content)
        if placeholders:
            issues.append(f"Found {len(placeholders)} reference placeholders: {', '.join(placeholders[:5])}...")
        
        # Check for modified citation codes
        citations = re.findall(r'\\citep?\{([^}]+)\}', content)
        for cite in citations:
            if '_modified' in cite or 'CHANGE' in cite:
                issues.append(f"Modified citation found: {cite}")
        
        # Check for missing citations after statements
        sentences = re.split(r'[.!?]\s+', content)
        for sent in sentences:
            if any(keyword in sent.lower() for keyword in 
                   ['studies have shown', 'research demonstrates', 'it has been reported',
                    'previous work', 'prior research']):
                if not re.search(r'\\citep?\{', sent):
                    issues.append(f"Statement may need citation: '{sent[:50]}...'")
        
        return issues
    
    def validate_document(self, target: str = 'manuscript'):
        """Run all validations on document."""
        print(f"\nValidating {target} document...")
        print("=" * 60)
        
        target_dir = self.root_dir / self.config['directories'][target]
        sections_dir = target_dir / 'sections'
        figures_dir = self.root_dir / self.config['directories']['figures']
        
        all_issues = {}
        
        # Validate abstract
        abstract_file = sections_dir / 'abstract.tex'
        if abstract_file.exists():
            issues = self.validate_abstract(abstract_file)
            if issues:
                all_issues['Abstract'] = issues
        
        # Validate introduction
        intro_file = sections_dir / 'introduction.tex'
        if intro_file.exists():
            issues = self.validate_introduction(intro_file)
            if issues:
                all_issues['Introduction'] = issues
        
        # Validate figures
        issues = self.validate_figures(figures_dir)
        if issues:
            all_issues['Figures'] = issues
        
        # Validate statistics in results
        results_file = sections_dir / 'results.tex'
        if results_file.exists():
            issues = self.validate_statistics(results_file)
            if issues:
                all_issues['Statistics'] = issues
        
        # Check references
        for tex_file in sections_dir.glob('*.tex'):
            issues = self.check_references(tex_file)
            if issues:
                all_issues[f'References ({tex_file.name})'] = issues
        
        # Report results
        if all_issues:
            print("\n❗ Validation Issues Found:\n")
            for section, issues in all_issues.items():
                print(f"{section}:")
                for issue in issues:
                    print(f"  • {issue}")
                print()
        else:
            print("\n✓ All validations passed!")
        
        return all_issues
    
    def create_figure_script(self, fig_name: str, fig_type: str = 'line'):
        """Create figure script following standards."""
        figures_dir = self.root_dir / self.config['directories']['figures']
        figures_dir.mkdir(exist_ok=True)
        
        templates = {
            'line': self._get_line_figure_template(),
            'bar': self._get_bar_figure_template(),
            'heatmap': self._get_heatmap_figure_template(),
        }
        
        if fig_type not in templates:
            print(f"Unknown figure type: {fig_type}")
            return
        
        script_file = figures_dir / f"{fig_name}.py"
        if script_file.exists():
            response = input(f"{fig_name}.py exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                return
        
        content = templates[fig_type].format(fig_name=fig_name)
        script_file.write_text(content)
        print(f"✓ Created {fig_name}.py with scientific standards")
    
    def _get_line_figure_template(self) -> str:
        """Line plot template with standards."""
        return '''#!/usr/bin/env python3
"""
Figure: {fig_name}
Following scientific figure standards.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Generate example data (replace with your data)
x = np.linspace(0, 10, 100)
y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
y2 = np.cos(x) + np.random.normal(0, 0.1, 100)

# Plot data
ax.plot(x, y1, label='Condition 1', linewidth=2)
ax.plot(x, y2, label='Condition 2', linewidth=2)

# REQUIRED: Set axis labels with units
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Amplitude (μV)', fontsize=12)

# REQUIRED: Set appropriate ticks (3-5 per axis)
ax.set_xticks(np.linspace(0, 10, 5))
ax.set_yticks(np.linspace(-2, 2, 5))

# Add legend
ax.legend(loc='best', framealpha=0.9)

# Add title
ax.set_title('{fig_name}', fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('{fig_name}.pdf', dpi=300, bbox_inches='tight')
plt.savefig('{fig_name}.png', dpi=300, bbox_inches='tight')

plt.show()
'''
    
    def _get_bar_figure_template(self) -> str:
        """Bar plot template with standards."""
        return '''#!/usr/bin/env python3
"""
Figure: {fig_name}
Bar plot following scientific standards.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set random seed
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Generate example data (replace with your data)
groups = ['Control', 'Treatment A', 'Treatment B']
means = [1.0, 1.5, 2.0]
sems = [0.1, 0.15, 0.12]
n_samples = [20, 20, 20]

# Create bar plot
x_pos = np.arange(len(groups))
bars = ax.bar(x_pos, means, yerr=sems, capsize=5, alpha=0.8)

# REQUIRED: Set y-axis to start from 0 for bar plots
ax.set_ylim(0, max(means) * 1.3)

# REQUIRED: Set axis labels with units
ax.set_xlabel('Condition', fontsize=12)
ax.set_ylabel('Response Rate (Hz)', fontsize=12)

# Set x-tick labels
ax.set_xticks(x_pos)
ax.set_xticklabels(groups)

# Add sample sizes
for i, (bar, n) in enumerate(zip(bars, n_samples)):
    ax.text(bar.get_x() + bar.get_width()/2, 0.05, f'n={n}', 
            ha='center', va='bottom', fontsize=10)

# Add significance markers (example)
# ax.plot([0, 1], [2.3, 2.3], 'k-', linewidth=1)
# ax.text(0.5, 2.35, '*', ha='center', fontsize=14)

# Add title
ax.set_title('{fig_name}', fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('{fig_name}.pdf', dpi=300, bbox_inches='tight')
plt.savefig('{fig_name}.png', dpi=300, bbox_inches='tight')

plt.show()
'''
    
    def _get_heatmap_figure_template(self) -> str:
        """Heatmap template with standards."""
        return '''#!/usr/bin/env python3
"""
Figure: {fig_name}
Heatmap following scientific standards.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Generate example data (replace with your data)
data = np.random.randn(10, 12) * 2 + 5

# Create heatmap
sns.heatmap(data, 
            cmap='RdBu_r',
            center=5,
            vmin=0, vmax=10,
            cbar_kws={{'label': 'Power (dB)'}},
            ax=ax)

# REQUIRED: Set axis labels with units
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)

# Set appropriate ticks
freq_labels = [f'{{i*10}}' for i in range(12)]
time_labels = [f'{{i*0.1:.1f}}' for i in range(10)]
ax.set_xticklabels(freq_labels[::3])  # Show every 3rd label
ax.set_yticklabels(time_labels[::2])   # Show every 2nd label

# Add title
ax.set_title('{fig_name}', fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('{fig_name}.pdf', dpi=300, bbox_inches='tight')
plt.savefig('{fig_name}.png', dpi=300, bbox_inches='tight')

plt.show()
'''
    
    def _load_config(self) -> dict:
        """Load configuration."""
        config_path = self.root_dir / self.config_file
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        
        return {
            'project': {'name': 'Research Project', 'version': '0.1.0'},
            'directories': {
                'manuscript': 'manuscript',
                'figures': 'figures',
                'tables': 'tables',
                'build': 'build'
            }
        }


def main():
    """Main entry point for scientific features."""
    parser = argparse.ArgumentParser(
        description='SciTeX Scientific - Paper Management with Best Practices'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Validation command
    val_parser = subparsers.add_parser('validate', 
                                      help='Validate document against guidelines')
    val_parser.add_argument('target', nargs='?', default='manuscript')
    
    # Create section with templates
    create_parser = subparsers.add_parser('create-section', 
                                         help='Create section with template')
    create_parser.add_argument('section', 
                              choices=['abstract', 'introduction', 'methods', 
                                      'results', 'discussion'])
    create_parser.add_argument('--target', default='manuscript')
    
    # Create figure script
    fig_parser = subparsers.add_parser('create-figure', 
                                      help='Create figure script')
    fig_parser.add_argument('name', help='Figure name')
    fig_parser.add_argument('--type', choices=['line', 'bar', 'heatmap'], 
                           default='line')
    
    args = parser.parse_args()
    
    # Initialize
    scitex = SciTeXScientific()
    
    if args.command == 'validate':
        scitex.validate_document(args.target)
    elif args.command == 'create-section':
        scitex.create_section_template(args.section, args.target)
    elif args.command == 'create-figure':
        scitex.create_figure_script(args.name, args.type)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()