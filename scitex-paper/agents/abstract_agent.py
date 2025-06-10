#!/usr/bin/env python3
"""
Abstract Agent: Specialized handler for scientific abstracts
Ensures proper 7-section structure and word limits.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class AbstractAgent:
    """Agent specialized in abstract writing and validation."""
    
    def __init__(self):
        self.required_sections = [
            "Basic Introduction",
            "Detailed Background", 
            "General Problem",
            "Main Result",
            "Results with Comparisons",
            "General Context",
            "Broader Perspective"
        ]
        self.word_limits = (150, 250)
        
    def create_template(self, project_info: Dict[str, str]) -> str:
        """Create abstract template with project-specific hints."""
        return f"""% Abstract ({self.word_limits[0]}-{self.word_limits[1]} words, 7 sections)
\\begin{{abstract}}

% 1. Basic Introduction (1-2 sentences)
% Introduce the general field and fundamental concepts
[Your field] represents a fundamental aspect of [domain], with [key concept] serving as...

% 2. Detailed Background (2-3 sentences)  
% Provide specific context about your research area
{project_info.get('background_hint', '[Specific technique/phenomenon]')} has been shown to...
Previous studies have demonstrated...
However, [limitation/gap]...

% 3. General Problem (1 sentence)
% State the key challenge your work addresses
Current methods cannot [specific limitation], limiting [broader impact].

% 4. Main Result (1 sentence with "Here we show...")
% State your primary finding/contribution
Here we show that {project_info.get('method_hint', '[your approach]')} enables [key achievement].

% 5. Results with Comparisons (2-3 sentences)
% Provide specific results with quantitative comparisons
Our method achieves [metric] of [value], compared to [baseline value] for existing approaches.
Validation on [datasets/experiments] demonstrated [key finding].
[Additional key result].

% 6. General Context (1-2 sentences)
% Place your work in the broader scientific context
This advance enables [new capability], facilitating [broader application].

% 7. Broader Perspective (2-3 sentences)
% Discuss implications and future impact
By [key contribution], our work [impact on field].
The approach [generalizability/future potential].

\\end{{abstract}}"""
    
    def validate(self, abstract_text: str) -> Dict[str, List[str]]:
        """Validate abstract structure and content."""
        issues = {"structure": [], "content": [], "style": []}
        
        # Remove LaTeX commands for analysis
        clean_text = self._clean_latex(abstract_text)
        
        # Check word count
        word_count = len(clean_text.split())
        if word_count < self.word_limits[0]:
            issues["content"].append(
                f"Too short ({word_count} words). Should be {self.word_limits[0]}-{self.word_limits[1]} words."
            )
        elif word_count > self.word_limits[1]:
            issues["content"].append(
                f"Too long ({word_count} words). Should be {self.word_limits[0]}-{self.word_limits[1]} words."
            )
        
        # Check for "Here we show"
        if "here we show" not in clean_text.lower():
            issues["structure"].append("Missing 'Here we show...' statement (Section 4)")
        
        # Check for comparisons
        comparison_words = ["compared", "versus", "than", "baseline", "existing", "previous"]
        if not any(word in clean_text.lower() for word in comparison_words):
            issues["content"].append("Missing comparison with previous work (Section 5)")
        
        # Check sentence count (rough proxy for 7 sections)
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 7:
            issues["structure"].append(
                f"Only {len(sentences)} sentences. Should have ~7-10 for proper structure."
            )
        
        # Style checks
        if re.search(r'\b(very|really|quite|extremely)\b', clean_text, re.I):
            issues["style"].append("Avoid intensifiers (very, really, quite, extremely)")
        
        if re.search(r'\b(we|our)\b', clean_text[:50], re.I):
            issues["style"].append("Avoid first person in opening sentences")
        
        return issues
    
    def suggest_improvements(self, abstract_text: str) -> List[str]:
        """Suggest specific improvements for the abstract."""
        suggestions = []
        clean_text = self._clean_latex(abstract_text)
        
        # Check opening
        first_sentence = clean_text.split('.')[0]
        if len(first_sentence.split()) < 10:
            suggestions.append(
                "Opening sentence is too short. Start with broader context."
            )
        
        # Check for quantitative results
        if not re.search(r'\d+', clean_text):
            suggestions.append(
                "Add quantitative results (percentages, fold-improvements, etc.)"
            )
        
        # Check for future/impact statement
        if not any(word in clean_text.lower() for word in 
                  ["enable", "facilitate", "accelerate", "improve", "advance"]):
            suggestions.append(
                "Add impact statement: how does this enable future work?"
            )
        
        return suggestions
    
    def extract_sections(self, abstract_text: str) -> Dict[int, str]:
        """Extract the 7 sections from abstract text."""
        sections = {}
        
        # Look for section markers in comments
        pattern = r'%\s*(\d+)\.\s*([^(]+)\([^)]+\)\s*\n([^%]+)'
        matches = re.finditer(pattern, abstract_text, re.MULTILINE)
        
        for match in matches:
            section_num = int(match.group(1))
            section_text = match.group(3).strip()
            sections[section_num] = section_text
        
        return sections
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands for text analysis."""
        # Remove comments
        text = re.sub(r'%.*?\n', ' ', text)
        # Remove commands
        text = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', ' ', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
        # Remove environment markers
        text = re.sub(r'\\begin\{[^}]+\}', '', text)
        text = re.sub(r'\\end\{[^}]+\}', '', text)
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def score_abstract(self, abstract_text: str) -> Dict[str, float]:
        """Score abstract quality (0-100)."""
        scores = {
            "structure": 100,
            "content": 100,
            "clarity": 100,
            "impact": 100
        }
        
        issues = self.validate(abstract_text)
        
        # Structure score
        scores["structure"] -= len(issues["structure"]) * 20
        
        # Content score
        scores["content"] -= len(issues["content"]) * 15
        
        # Clarity score
        scores["clarity"] -= len(issues["style"]) * 10
        
        # Impact score (check for impact words)
        clean_text = self._clean_latex(abstract_text)
        impact_words = ["novel", "first", "enable", "advance", "breakthrough", 
                       "significant", "substantial", "important"]
        impact_count = sum(1 for word in impact_words if word in clean_text.lower())
        scores["impact"] = min(100, impact_count * 20)
        
        # Ensure scores are non-negative
        scores = {k: max(0, v) for k, v in scores.items()}
        
        return scores


def main():
    """Example usage."""
    agent = AbstractAgent()
    
    # Create template
    project_info = {
        "background_hint": "Phase-amplitude coupling (PAC)",
        "method_hint": "GPU acceleration"
    }
    
    template = agent.create_template(project_info)
    print("Abstract Template:")
    print(template)
    print("\n" + "="*60 + "\n")
    
    # Example validation
    example_abstract = """
    \\begin{abstract}
    Neural oscillations are important. PAC is a useful measure.
    However, it is slow. Here we show that GPUs make it faster.
    It's 100x faster. This is good for neuroscience.
    \\end{abstract}
    """
    
    issues = agent.validate(example_abstract)
    print("Validation Issues:")
    for category, items in issues.items():
        if items:
            print(f"\n{category.title()}:")
            for item in items:
                print(f"  • {item}")


if __name__ == "__main__":
    main()