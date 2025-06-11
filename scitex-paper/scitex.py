#!/usr/bin/env python3
"""
SciTeX: Modern Scientific Paper Management System
A flexible, Python-based LaTeX manuscript builder for scientific publications.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. Using JSON for configuration.")


class SciTeX:
    """Main class for scientific paper management."""
    
    def __init__(self, root_dir: Optional[Path] = None, config_file: str = "scitex.yaml"):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from YAML or JSON."""
        config_path = self.root_dir / self.config_file
        
        # Try YAML first
        if config_path.with_suffix('.yaml').exists():
            config_path = config_path.with_suffix('.yaml')
            if YAML_AVAILABLE:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        
        # Try JSON
        if config_path.with_suffix('.json').exists():
            config_path = config_path.with_suffix('.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # No config found
        return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'project': {
                'name': 'New Project',
                'title': 'Paper Title',
                'version': '0.1.0',
            },
            'directories': {
                'manuscript': 'manuscript',
                'revision': 'revision',
                'supplementary': 'supplementary',
                'figures': 'figures',
                'tables': 'tables',
                'build': 'build',
                'templates': 'templates',
            },
            'latex': {
                'engine': 'pdflatex',
                'bibtex': 'bibtex',
                'options': ['-interaction=nonstopmode'],
            },
            'features': {
                'clean_build': True,
                'version_output': True,
                'open_after_compile': False,
            }
        }
    
    def init(self, project_name: Optional[str] = None, journal: Optional[str] = None):
        """Initialize a new paper project."""
        print(f"Initializing SciTeX paper project in {self.root_dir}")
        
        # Update config with project name
        if project_name:
            self.config['project']['name'] = project_name
            self.config['project']['title'] = project_name
        
        # Create directory structure
        dirs = self.config['directories']
        for dir_type, dir_name in dirs.items():
            dir_path = self.root_dir / dir_name
            dir_path.mkdir(exist_ok=True, parents=True)
            print(f"  ✓ Created {dir_name}/")
        
        # Create config file
        config_path = self.root_dir / self.config_file
        self._save_config(config_path)
        print(f"  ✓ Created {self.config_file}")
        
        # Create main scitex.py
        scitex_path = self.root_dir / 'scitex.py'
        if not scitex_path.exists():
            shutil.copy2(__file__, scitex_path)
            print("  ✓ Created scitex.py")
        
        # Create templates
        self._create_templates()
        
        # Create example manuscript structure
        self._create_manuscript_structure()
        
        print("\n✓ SciTeX project initialized successfully!")
        print(f"\nNext steps:")
        print(f"  1. Edit {self.config_file} to configure your project")
        print(f"  2. Edit manuscript/main.tex and section files")
        print(f"  3. Run: python scitex.py compile")
    
    def _save_config(self, path: Path):
        """Save configuration to file."""
        if YAML_AVAILABLE and path.suffix == '.yaml':
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        else:
            # Save as JSON
            path = path.with_suffix('.json')
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def _create_templates(self):
        """Create template files."""
        template_dir = self.root_dir / self.config['directories']['templates']
        template_dir.mkdir(exist_ok=True)
        
        # Section templates
        templates = {
            'abstract': """\\begin{abstract}
Your abstract here. Should be concise and informative, typically 150-250 words.
\\end{abstract}""",
            
            'introduction': """\\section{Introduction}
\\label{sec:introduction}

Start your introduction here. Set the context and importance of your work.

\\subsection{Background}
Provide relevant background information.

\\subsection{Motivation}
Explain the motivation for your research.

\\subsection{Contributions}
List the main contributions of your work:
\\begin{itemize}
    \\item First contribution
    \\item Second contribution
\\end{itemize}""",
            
            'methods': """\\section{Methods}
\\label{sec:methods}

Describe your methodology in detail.

\\subsection{Data Collection}
Explain how data was collected or generated.

\\subsection{Analysis Approach}
Detail your analysis methods.""",
            
            'results': """\\section{Results}
\\label{sec:results}

Present your findings clearly and objectively.

\\subsection{Main Findings}
Describe your primary results.

\\subsection{Statistical Analysis}
Include relevant statistical analyses.""",
            
            'discussion': """\\section{Discussion}
\\label{sec:discussion}

Interpret your results and discuss their implications.

\\subsection{Interpretation}
What do your results mean?

\\subsection{Limitations}
Acknowledge any limitations of your study.

\\subsection{Future Work}
Suggest directions for future research.""",
        }
        
        for name, content in templates.items():
            template_path = template_dir / f"{name}_template.tex"
            if not template_path.exists():
                template_path.write_text(content)
                print(f"  ✓ Created template: {name}_template.tex")
    
    def _create_manuscript_structure(self):
        """Create basic manuscript structure."""
        manuscript_dir = self.root_dir / self.config['directories']['manuscript']
        sections_dir = manuscript_dir / 'sections'
        sections_dir.mkdir(exist_ok=True)
        
        # Create main.tex
        main_tex = manuscript_dir / 'main.tex'
        if not main_tex.exists():
            main_content = """\\documentclass[preprint,12pt]{elsarticle}

% Packages
\\usepackage{graphicx}
\\usepackage{amsmath,amssymb}
\\usepackage{hyperref}
\\usepackage{cleveref}

\\begin{document}

\\begin{frontmatter}
    \\title{""" + self.config['project']['title'] + """}
    
    \\author{First Author}
    \\author{Second Author}
    
    \\begin{abstract}
    Abstract goes here.
    \\end{abstract}
    
    \\begin{keyword}
    keyword1 \\sep keyword2 \\sep keyword3
    \\end{keyword}
\\end{frontmatter}

% Sections
\\input{sections/introduction}
\\input{sections/methods}
\\input{sections/results}
\\input{sections/discussion}

% Bibliography
\\bibliography{references}

\\end{document}"""
            main_tex.write_text(main_content)
            print("  ✓ Created manuscript/main.tex")
        
        # Create empty section files
        sections = ['introduction', 'methods', 'results', 'discussion']
        for section in sections:
            section_file = sections_dir / f"{section}.tex"
            if not section_file.exists():
                # Copy from template if exists
                template = self.root_dir / self.config['directories']['templates'] / f"{section}_template.tex"
                if template.exists():
                    shutil.copy2(template, section_file)
                else:
                    section_file.write_text(f"\\section{{{section.title()}}}\n\n% Add content here\n")
                print(f"  ✓ Created sections/{section}.tex")
        
        # Create empty bibliography
        bib_file = manuscript_dir / 'references.bib'
        if not bib_file.exists():
            bib_file.write_text("% Add your references here\n")
            print("  ✓ Created references.bib")
    
    def compile(self, target: str = 'manuscript', clean: bool = None, 
                view: bool = None, verbose: bool = False):
        """Compile LaTeX document."""
        print(f"\nCompiling {target}...")
        
        target_dir = self.root_dir / self.config['directories'].get(target, target)
        if not target_dir.exists():
            print(f"Error: {target_dir} does not exist")
            return False
        
        main_file = target_dir / 'main.tex'
        if not main_file.exists():
            print(f"Error: {main_file} not found")
            return False
        
        # Use config defaults if not specified
        if clean is None:
            clean = self.config['features'].get('clean_build', True)
        if view is None:
            view = self.config['features'].get('open_after_compile', False)
        
        # Save current directory
        original_dir = os.getcwd()
        os.chdir(target_dir)
        
        try:
            # Clean if requested
            if clean:
                self._clean_aux_files()
            
            # Compile
            success = self._run_latex_compilation(verbose)
            
            if success:
                print("✓ Compilation successful!")
                
                # Version output
                if self.config['features'].get('version_output', True):
                    self._version_output(target)
                
                # Open PDF
                if view:
                    self._open_pdf(Path('main.pdf'))
            else:
                print("✗ Compilation failed!")
            
            return success
            
        finally:
            os.chdir(original_dir)
    
    def _run_latex_compilation(self, verbose: bool = False) -> bool:
        """Run the actual LaTeX compilation."""
        engine = self.config['latex']['engine']
        options = self.config['latex']['options'].copy()
        
        if not verbose:
            options.append('-quiet')
        
        try:
            # First pass
            cmd = [engine] + options + ['main.tex']
            subprocess.run(cmd, check=True, capture_output=not verbose)
            
            # Check for bibliography
            if Path('references.bib').exists() or Path('main.bib').exists():
                bibtex = self.config['latex']['bibtex']
                subprocess.run([bibtex, 'main'], check=True, capture_output=not verbose)
                
                # Two more passes for citations
                subprocess.run(cmd, check=True, capture_output=not verbose)
                subprocess.run(cmd, check=True, capture_output=not verbose)
            
            return True
            
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Error: {e}")
            return False
    
    def _clean_aux_files(self):
        """Clean auxiliary files."""
        extensions = ['.aux', '.log', '.out', '.toc', '.bbl', '.blg', 
                     '.fls', '.fdb_latexmk', '.synctex.gz', '.nav', '.snm']
        
        cleaned = 0
        for ext in extensions:
            for file in Path('.').glob(f'*{ext}'):
                file.unlink()
                cleaned += 1
        
        if cleaned > 0:
            print(f"  Cleaned {cleaned} auxiliary files")
    
    def _version_output(self, target: str):
        """Create versioned output."""
        pdf_file = Path('main.pdf')
        if not pdf_file.exists():
            return
        
        build_dir = self.root_dir / self.config['directories']['build']
        build_dir.mkdir(exist_ok=True)
        
        # Create versioned filename
        version = self.config['project'].get('version', '0.1.0')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_name = f"{target}_v{version}_{timestamp}.pdf"
        output_path = build_dir / output_name
        
        shutil.copy2(pdf_file, output_path)
        print(f"  ✓ Saved to: build/{output_name}")
        
        # Update latest symlink
        latest_link = build_dir / f"{target}_latest.pdf"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_name)
    
    def _open_pdf(self, pdf_path: Path):
        """Open PDF with system viewer."""
        if not pdf_path.exists():
            return
        
        if sys.platform == 'darwin':
            subprocess.run(['open', str(pdf_path)])
        elif sys.platform == 'linux':
            subprocess.run(['xdg-open', str(pdf_path)])
        elif sys.platform == 'win32':
            subprocess.run(['start', '', str(pdf_path)], shell=True)
    
    def create(self, section: str, target: str = 'manuscript'):
        """Create a new section from template."""
        target_dir = self.root_dir / self.config['directories'][target]
        sections_dir = target_dir / 'sections'
        sections_dir.mkdir(exist_ok=True)
        
        section_file = sections_dir / f"{section}.tex"
        if section_file.exists():
            response = input(f"{section}.tex already exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                return
        
        # Check for template
        template_dir = self.root_dir / self.config['directories']['templates']
        template_file = template_dir / f"{section}_template.tex"
        
        if template_file.exists():
            shutil.copy2(template_file, section_file)
            print(f"✓ Created {section}.tex from template")
        else:
            # Create basic content
            content = f"""\\section{{{section.replace('_', ' ').title()}}}
\\label{{sec:{section}}}

% Add your content here

"""
            section_file.write_text(content)
            print(f"✓ Created {section}.tex")
    
    def wordcount(self, target: str = 'manuscript', detailed: bool = False):
        """Count words in document."""
        target_dir = self.root_dir / self.config['directories'][target]
        main_file = target_dir / 'main.tex'
        
        if not main_file.exists():
            print(f"Error: {main_file} not found")
            return
        
        # Try texcount first
        try:
            cmd = ['texcount']
            if not detailed:
                cmd.append('-brief')
            cmd.append(str(main_file))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            
        except FileNotFoundError:
            # Fallback to simple count
            print("texcount not found. Using simple word count...")
            self._simple_wordcount(main_file)
    
    def _simple_wordcount(self, tex_file: Path):
        """Simple word count implementation."""
        word_count = 0
        
        # Read main file and included files
        content = tex_file.read_text()
        
        # Remove comments
        content = re.sub(r'%.*', '', content)
        
        # Remove LaTeX commands (simple approach)
        content = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^\}]*\})?', ' ', content)
        
        # Count words
        words = content.split()
        word_count = len(words)
        
        print(f"Approximate word count: {word_count}")
    
    def status(self):
        """Show project status."""
        print(f"\nSciTeX Project Status")
        print("=" * 50)
        print(f"Project: {self.config['project']['name']}")
        print(f"Version: {self.config['project']['version']}")
        print(f"Root: {self.root_dir}")
        
        # Check directories
        print("\nDirectories:")
        for dir_type, dir_name in self.config['directories'].items():
            dir_path = self.root_dir / dir_name
            exists = "✓" if dir_path.exists() else "✗"
            print(f"  {exists} {dir_name}/")
        
        # Check for PDFs
        build_dir = self.root_dir / self.config['directories']['build']
        if build_dir.exists():
            pdfs = list(build_dir.glob('*.pdf'))
            if pdfs:
                print(f"\nBuilt PDFs ({len(pdfs)}):")
                for pdf in sorted(pdfs)[-5:]:  # Show last 5
                    size = pdf.stat().st_size / 1024 / 1024  # MB
                    print(f"  - {pdf.name} ({size:.1f} MB)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SciTeX - Modern Scientific Paper Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scitex.py init --project "My Paper"     # Initialize new project
  python scitex.py compile                        # Compile manuscript
  python scitex.py compile -v                     # Compile and view PDF
  python scitex.py create results                 # Create results section
  python scitex.py wordcount                      # Show word count
  python scitex.py status                         # Show project status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize new paper project')
    init_parser.add_argument('--project', '-p', help='Project name')
    init_parser.add_argument('--journal', '-j', help='Target journal')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile document')
    compile_parser.add_argument('target', nargs='?', default='manuscript',
                               help='Target to compile (default: manuscript)')
    compile_parser.add_argument('--clean', '-c', action='store_true',
                               help='Clean auxiliary files before compilation')
    compile_parser.add_argument('--view', '-v', action='store_true',
                               help='Open PDF after compilation')
    compile_parser.add_argument('--verbose', action='store_true',
                               help='Show detailed compilation output')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new section')
    create_parser.add_argument('section', help='Section name (e.g., methods, results)')
    create_parser.add_argument('--target', '-t', default='manuscript',
                              help='Target document (default: manuscript)')
    
    # Word count command
    wc_parser = subparsers.add_parser('wordcount', help='Count words in document')
    wc_parser.add_argument('target', nargs='?', default='manuscript',
                          help='Target to count (default: manuscript)')
    wc_parser.add_argument('--detailed', '-d', action='store_true',
                          help='Show detailed breakdown')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show project status')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize SciTeX
    scitex = SciTeX()
    
    # Execute command
    if args.command == 'init':
        scitex.init(project_name=args.project, journal=args.journal)
    
    elif args.command == 'compile':
        scitex.compile(target=args.target, clean=args.clean, 
                      view=args.view, verbose=args.verbose)
    
    elif args.command == 'create':
        scitex.create(section=args.section, target=args.target)
    
    elif args.command == 'wordcount':
        scitex.wordcount(target=args.target, detailed=args.detailed)
    
    elif args.command == 'status':
        scitex.status()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()