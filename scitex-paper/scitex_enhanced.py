#!/usr/bin/env python3
"""
SciTeX Enhanced: Complete Scientific Paper Management System
Includes BibTeX, figure/table management, diff support, and source linking.
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
from typing import Dict, List, Optional, Tuple, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class SciTeXEnhanced:
    """Enhanced scientific paper management with full LaTeX compilation support."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.config_file = "scitex.yaml"
        self.config = self._load_config()
        
    def compile_full(self, target: str = 'manuscript', **kwargs):
        """Full compilation with BibTeX, multiple passes, and cross-references."""
        target_dir = self.root_dir / self.config['directories'][target]
        build_dir = self.root_dir / self.config['directories']['build']
        build_dir.mkdir(exist_ok=True)
        
        # Save current directory
        original_dir = os.getcwd()
        os.chdir(target_dir)
        
        try:
            print(f"Starting full compilation of {target}...")
            
            # Step 1: Initial LaTeX compilation
            print("  Pass 1: Initial compilation...")
            if not self._run_latex():
                return False
            
            # Step 2: BibTeX processing
            if self._has_bibliography():
                print("  Running BibTeX...")
                if not self._run_bibtex():
                    print("  Warning: BibTeX failed, continuing...")
            
            # Step 3: Multiple passes for cross-references
            print("  Pass 2: Updating citations...")
            self._run_latex()
            
            print("  Pass 3: Finalizing references...")
            self._run_latex()
            
            # Step 4: Check if another pass is needed
            if self._needs_rerun():
                print("  Pass 4: Final adjustments...")
                self._run_latex()
            
            # Step 5: Post-processing
            self._post_process(target, build_dir)
            
            print("✓ Compilation completed successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Compilation failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def _run_latex(self) -> bool:
        """Run LaTeX engine."""
        engine = self.config['latex']['engine']
        options = self.config['latex']['options'].copy()
        
        cmd = [engine] + options + ['main.tex']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for errors in log
            if result.returncode != 0:
                self._parse_latex_errors('main.log')
                return False
            
            return True
        except Exception as e:
            print(f"  LaTeX error: {e}")
            return False
    
    def _run_bibtex(self) -> bool:
        """Run BibTeX processor."""
        bibtex = self.config['latex'].get('bibtex', 'bibtex')
        
        try:
            cmd = [bibtex, 'main']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  BibTeX warnings: {result.stdout}")
            
            return True
        except Exception as e:
            print(f"  BibTeX error: {e}")
            return False
    
    def _has_bibliography(self) -> bool:
        """Check if document has bibliography."""
        # Check for .bib files
        bib_files = list(Path('.').glob('*.bib'))
        if bib_files:
            return True
        
        # Check for \bibliography command in .tex files
        for tex_file in Path('.').glob('**/*.tex'):
            content = tex_file.read_text()
            if r'\bibliography' in content or r'\addbibresource' in content:
                return True
        
        return False
    
    def _needs_rerun(self) -> bool:
        """Check if LaTeX needs another run."""
        log_file = Path('main.log')
        if not log_file.exists():
            return False
        
        log_content = log_file.read_text()
        
        # Check for rerun warnings
        rerun_patterns = [
            r'Rerun to get cross-references right',
            r'Rerun LaTeX',
            r'Label\(s\) may have changed'
        ]
        
        for pattern in rerun_patterns:
            if re.search(pattern, log_content):
                return True
        
        return False
    
    def _parse_latex_errors(self, log_file: str):
        """Parse and display LaTeX errors."""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            errors = []
            for i, line in enumerate(lines):
                if line.startswith('!'):
                    # Found an error
                    error_msg = line[1:].strip()
                    
                    # Try to find line number
                    for j in range(i+1, min(i+5, len(lines))):
                        if lines[j].startswith('l.'):
                            error_msg += f" at {lines[j].strip()}"
                            break
                    
                    errors.append(error_msg)
            
            if errors:
                print("\n  LaTeX Errors:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"    - {error}")
                    
        except Exception:
            pass
    
    def _post_process(self, target: str, build_dir: Path):
        """Post-process compiled document."""
        # Copy PDF to build directory with versioning
        pdf_file = Path('main.pdf')
        if pdf_file.exists():
            # Create versioned filename
            version = self.config['project'].get('version', '0.1.0')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if self.config['output'].get('timestamp', True):
                output_name = f"{target}_v{version}_{timestamp}.pdf"
            else:
                output_name = f"{target}_v{version}.pdf"
            
            output_path = build_dir / output_name
            shutil.copy2(pdf_file, output_path)
            
            # Create/update latest symlink
            latest_link = build_dir / f"{target}_latest.pdf"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(output_name)
            
            print(f"\n  Output saved to: build/{output_name}")
    
    def manage_figures(self, action: str = 'list'):
        """Manage figures in the document."""
        figures_dir = self.root_dir / self.config['directories']['figures']
        figures_dir.mkdir(exist_ok=True)
        
        if action == 'list':
            # List all figures
            print("\nFigures in project:")
            for fig_type in ['*.pdf', '*.png', '*.jpg', '*.eps']:
                for fig in figures_dir.glob(fig_type):
                    size = fig.stat().st_size / 1024  # KB
                    print(f"  - {fig.name} ({size:.1f} KB)")
            
            # Check usage in .tex files
            self._check_figure_usage()
            
        elif action == 'unused':
            # Find unused figures
            used_figures = self._get_used_figures()
            all_figures = set(f.name for f in figures_dir.glob('*') 
                            if f.suffix in ['.pdf', '.png', '.jpg', '.eps'])
            
            unused = all_figures - used_figures
            if unused:
                print("\nUnused figures:")
                for fig in sorted(unused):
                    print(f"  - {fig}")
            else:
                print("\nAll figures are used.")
    
    def _get_used_figures(self) -> set:
        """Get set of figures used in .tex files."""
        used = set()
        
        # Search in all .tex files
        for tex_file in self.root_dir.rglob('*.tex'):
            content = tex_file.read_text()
            
            # Find \includegraphics commands
            pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
            for match in re.finditer(pattern, content):
                fig_path = match.group(1)
                # Extract filename
                fig_name = Path(fig_path).name
                used.add(fig_name)
        
        return used
    
    def _check_figure_usage(self):
        """Check which figures are used in the document."""
        used_figures = self._get_used_figures()
        
        if used_figures:
            print(f"\nUsed figures ({len(used_figures)}):")
            for fig in sorted(used_figures):
                print(f"  ✓ {fig}")
    
    def manage_tables(self, action: str = 'list'):
        """Manage tables in the document."""
        tables_dir = self.root_dir / self.config['directories']['tables']
        tables_dir.mkdir(exist_ok=True)
        
        if action == 'list':
            print("\nTables in project:")
            
            # List .tex table files
            for table_file in tables_dir.glob('*.tex'):
                print(f"  - {table_file.name}")
            
            # Count tables in main document
            table_count = self._count_tables()
            print(f"\nTotal tables in document: {table_count}")
    
    def _count_tables(self) -> int:
        """Count tables in the document."""
        count = 0
        
        for tex_file in self.root_dir.rglob('*.tex'):
            content = tex_file.read_text()
            
            # Count \begin{table} environments
            count += len(re.findall(r'\\begin\{table', content))
        
        return count
    
    def create_diff(self, old_version: str, new_version: str = 'current'):
        """Create a diff between two versions using latexdiff."""
        print(f"Creating diff between {old_version} and {new_version}...")
        
        # Check if latexdiff is available
        if not shutil.which('latexdiff'):
            print("Error: latexdiff not found. Please install it.")
            return False
        
        build_dir = self.root_dir / self.config['directories']['build']
        manuscript_dir = self.root_dir / self.config['directories']['manuscript']
        
        # Find old version
        if old_version == 'latest':
            old_pdf = build_dir / 'manuscript_latest.pdf'
            old_tex = manuscript_dir / 'main.tex.backup'
        else:
            # Look for specific version
            old_candidates = list(build_dir.glob(f'*{old_version}*.pdf'))
            if not old_candidates:
                print(f"Error: Cannot find version {old_version}")
                return False
            old_pdf = old_candidates[0]
            old_tex = old_pdf.with_suffix('.tex')
        
        # Prepare new version
        if new_version == 'current':
            new_tex = manuscript_dir / 'main.tex'
        else:
            new_candidates = list(build_dir.glob(f'*{new_version}*.tex'))
            if not new_candidates:
                print(f"Error: Cannot find version {new_version}")
                return False
            new_tex = new_candidates[0]
        
        # Create diff
        diff_tex = manuscript_dir / 'diff.tex'
        
        try:
            cmd = ['latexdiff', str(old_tex), str(new_tex)]
            with open(diff_tex, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            
            print("✓ Diff created successfully!")
            
            # Compile diff
            print("Compiling diff document...")
            original_dir = os.getcwd()
            os.chdir(manuscript_dir)
            
            try:
                # Compile diff
                subprocess.run([self.config['latex']['engine'], 'diff.tex'], 
                             check=True, capture_output=True)
                
                # Move to build dir
                diff_pdf = Path('diff.pdf')
                if diff_pdf.exists():
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_name = f"diff_{old_version}_vs_{new_version}_{timestamp}.pdf"
                    shutil.copy2(diff_pdf, build_dir / output_name)
                    print(f"✓ Diff saved to: build/{output_name}")
                
            finally:
                os.chdir(original_dir)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error creating diff: {e}")
            return False
    
    def link_source(self, source_file: Path, target: str = 'manuscript'):
        """Link source code or data files to the manuscript."""
        # Create links directory
        target_dir = self.root_dir / self.config['directories'][target]
        links_dir = target_dir / 'links'
        links_dir.mkdir(exist_ok=True)
        
        # Create symlink
        link_name = links_dir / source_file.name
        
        if link_name.exists():
            link_name.unlink()
        
        link_name.symlink_to(source_file.absolute())
        print(f"✓ Linked {source_file} to {target}/links/")
        
        # Update .gitignore to exclude links
        gitignore = self.root_dir / '.gitignore'
        if gitignore.exists():
            content = gitignore.read_text()
            if 'links/' not in content:
                with open(gitignore, 'a') as f:
                    f.write('\n# Linked files\nlinks/\n')
    
    def package_submission(self, target: str = 'manuscript', 
                          anonymous: bool = False):
        """Package manuscript for journal submission."""
        print(f"Packaging {target} for submission...")
        
        # Create package directory
        timestamp = datetime.now().strftime('%Y%m%d')
        package_name = f"submission_{timestamp}"
        if anonymous:
            package_name += "_anonymous"
        
        package_dir = self.root_dir / 'submissions' / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy necessary files
        target_dir = self.root_dir / self.config['directories'][target]
        
        # Files to include
        include_patterns = [
            'main.tex',
            '*.bib',
            'sections/*.tex',
            '*.cls',
            '*.sty',
            '*.bst'
        ]
        
        for pattern in include_patterns:
            for file in target_dir.glob(pattern):
                dest = package_dir / file.relative_to(target_dir)
                dest.parent.mkdir(exist_ok=True)
                
                if anonymous and file.suffix == '.tex':
                    # Anonymize .tex files
                    self._copy_anonymized(file, dest)
                else:
                    shutil.copy2(file, dest)
        
        # Copy figures
        figures_dir = self.root_dir / self.config['directories']['figures']
        package_figs = package_dir / 'figures'
        package_figs.mkdir(exist_ok=True)
        
        used_figures = self._get_used_figures()
        for fig in used_figures:
            src = figures_dir / fig
            if src.exists():
                shutil.copy2(src, package_figs / fig)
        
        # Copy tables
        tables_dir = self.root_dir / self.config['directories']['tables']
        if tables_dir.exists():
            package_tables = package_dir / 'tables'
            shutil.copytree(tables_dir, package_tables, dirs_exist_ok=True)
        
        # Create ZIP archive
        archive_name = f"{package_name}.zip"
        shutil.make_archive(
            str(self.root_dir / 'submissions' / package_name),
            'zip',
            package_dir.parent,
            package_dir.name
        )
        
        print(f"✓ Submission package created: submissions/{archive_name}")
        
        # List contents
        print("\nPackage contents:")
        for file in sorted(package_dir.rglob('*')):
            if file.is_file():
                rel_path = file.relative_to(package_dir)
                print(f"  - {rel_path}")
    
    def _copy_anonymized(self, src: Path, dest: Path):
        """Copy file with anonymization."""
        content = src.read_text()
        
        # Remove author information
        content = re.sub(r'\\author\{[^}]+\}', r'\\author{Anonymous}', content)
        content = re.sub(r'\\affiliation\{[^}]+\}', r'\\affiliation{Anonymous Institution}', content)
        content = re.sub(r'\\email\{[^}]+\}', '', content)
        
        # Remove acknowledgments
        content = re.sub(r'\\section\*?\{Acknowledg', r'% \\section{Acknowledg', content)
        
        dest.write_text(content)
    
    def _load_config(self) -> dict:
        """Load configuration with defaults."""
        config_path = self.root_dir / self.config_file
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        
        # Return defaults
        return {
            'project': {
                'name': 'Research Project',
                'version': '0.1.0'
            },
            'directories': {
                'manuscript': 'manuscript',
                'revision': 'revision',
                'supplementary': 'supplementary',
                'figures': 'figures',
                'tables': 'tables',
                'build': 'build',
                'templates': 'templates'
            },
            'latex': {
                'engine': 'pdflatex',
                'bibtex': 'bibtex',
                'options': ['-interaction=nonstopmode']
            },
            'output': {
                'timestamp': True
            }
        }


def main():
    """Enhanced main entry point."""
    parser = argparse.ArgumentParser(
        description='SciTeX Enhanced - Complete Scientific Paper Management'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Compile command with full support
    compile_parser = subparsers.add_parser('compile', 
                                          help='Full LaTeX compilation with BibTeX')
    compile_parser.add_argument('target', nargs='?', default='manuscript')
    
    # Figure management
    fig_parser = subparsers.add_parser('figures', help='Manage figures')
    fig_parser.add_argument('action', choices=['list', 'unused'], 
                           default='list')
    
    # Table management
    table_parser = subparsers.add_parser('tables', help='Manage tables')
    table_parser.add_argument('action', choices=['list', 'count'], 
                             default='list')
    
    # Diff creation
    diff_parser = subparsers.add_parser('diff', help='Create version diff')
    diff_parser.add_argument('old', help='Old version')
    diff_parser.add_argument('new', nargs='?', default='current', 
                            help='New version')
    
    # Source linking
    link_parser = subparsers.add_parser('link', help='Link source files')
    link_parser.add_argument('source', type=Path, help='Source file to link')
    link_parser.add_argument('--target', default='manuscript')
    
    # Submission packaging
    package_parser = subparsers.add_parser('package', 
                                          help='Package for submission')
    package_parser.add_argument('--anonymous', action='store_true',
                               help='Anonymize for review')
    
    args = parser.parse_args()
    
    # Initialize enhanced SciTeX
    scitex = SciTeXEnhanced()
    
    if args.command == 'compile':
        scitex.compile_full(args.target)
    elif args.command == 'figures':
        scitex.manage_figures(args.action)
    elif args.command == 'tables':
        scitex.manage_tables(args.action)
    elif args.command == 'diff':
        scitex.create_diff(args.old, args.new)
    elif args.command == 'link':
        scitex.link_source(args.source, args.target)
    elif args.command == 'package':
        scitex.package_submission(anonymous=args.anonymous)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()