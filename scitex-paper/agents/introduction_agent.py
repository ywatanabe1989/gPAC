#!/usr/bin/env python3
"""
Introduction Agent: Specialized handler for scientific introductions
Ensures proper 8-section structure and logical flow.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class IntroductionAgent:
    """Agent specialized in introduction writing and validation."""
    
    def __init__(self):
        self.required_sections = [
            ("Opening Statement", 1, "Broad context that captures attention"),
            ("Importance of the Field", 2, "Why this area matters"),
            ("Existing Knowledge and Gaps", 3, "What we know and don't know (≥3 paragraphs)"),
            ("Limitations in Previous Works", 4, "Specific technical/methodological limitations"),
            ("Research Question or Hypothesis", 5, "Clear statement of what you're addressing"),
            ("Approach and Methods", 6, "Brief overview of your solution"),
            ("Overview of Results", 7, "Key findings preview (optional)"),
            ("Significance and Implications", 8, "Why this matters for the field")
        ]
        self.min_words = 1000
        
    def create_template(self, project_info: Dict[str, str]) -> str:
        """Create introduction template with project-specific content."""
        field = project_info.get('field', '[Your field]')
        problem = project_info.get('problem', '[key problem]')
        method = project_info.get('method', '[your approach]')
        
        template = f"""\\section{{Introduction}}
\\label{{sec:introduction}}

[START of 1. Opening Statement]
% Broad context that captures reader attention
{field} represents a fundamental challenge in [broader domain], with implications spanning from [application 1] to [application 2] \\citep{{}}. 
[END of 1. Opening Statement]

[START of 2. Importance of the Field]
% Why this area matters
Understanding {problem} is crucial for [reason 1] and [reason 2]. Recent advances in [related technology] have opened new possibilities for addressing these challenges, yet significant obstacles remain \\citep{{}}.
[END of 2. Importance of the Field]

[START of 3. Existing Knowledge and Gaps]
% What we know and don't know (≥3 paragraphs)
Extensive research has established that [established fact 1] \\citep{{}}. Studies by \\hlref{{Author1}} demonstrated [finding 1], while \\hlref{{Author2}} showed [finding 2]. These findings have been instrumental in [application].

However, current understanding remains limited in several key areas. First, [gap 1] remains poorly understood, with conflicting reports about [specific issue] \\citep{{}}. Second, [gap 2] has received little attention despite its potential importance for [application].

Furthermore, the relationship between [concept 1] and [concept 2] remains unclear. While some studies suggest [hypothesis 1] \\citep{{}}, others have found [contradictory evidence] \\citep{{}}, indicating the need for more comprehensive investigation.
[END of 3. Existing Knowledge and Gaps]

[START of 4. Limitations in Previous Works]
% Specific technical/methodological limitations
Previous approaches to {problem} have been constrained by several technical limitations. Traditional methods require [limitation 1], making them impractical for [modern application]. The computational complexity of [specific algorithm] scales as O(n²), rendering it infeasible for [large-scale scenario]. Additionally, existing solutions typically assume [unrealistic assumption], which rarely holds in real-world applications \\citep{{}}.
[END of 4. Limitations in Previous Works]

[START of 5. Research Question or Hypothesis]
% Clear statement of what you're addressing
We hypothesized that {method} could overcome these limitations by [key innovation], enabling [desired outcome] while maintaining [important property]. Specifically, we asked whether [specific question] could be achieved through [technical approach].
[END of 5. Research Question or Hypothesis]

[START of 6. Approach and Methods]
% Brief overview of your solution
To address these challenges, we developed {method}, a novel framework that leverages [key technology] to achieve [primary goal]. Our approach combines [technique 1] with [technique 2], enabling [capability 1] and [capability 2]. By [key innovation], we circumvent the limitations of previous methods while preserving [desirable properties].
[END of 6. Approach and Methods]

[START of 7. Overview of Results] % Optional section
% Key findings preview
Our experiments demonstrate that {method} achieves [quantitative improvement] compared to state-of-the-art methods, while requiring [reduced resource]. We further show that [additional finding], opening new possibilities for [application].
[END of 7. Overview of Results]

[START of 8. Significance and Implications]
% Why this matters for the field
This work represents a significant advance in {field}, with immediate applications in [domain 1] and [domain 2]. By removing the computational barriers that have limited [specific application], our approach enables researchers to [new capability]. More broadly, the principles underlying {method} may extend to [related problems], potentially transforming how we approach [general challenge].
[END of 8. Significance and Implications]"""
        
        return template
    
    def validate(self, intro_text: str) -> Dict[str, List[str]]:
        """Validate introduction structure and content."""
        issues = {
            "structure": [],
            "content": [],
            "citations": [],
            "flow": []
        }
        
        # Check for section markers
        for section_name, num, _ in self.required_sections:
            if num == 7:  # Optional section
                continue
            marker = f"[START of {num}. {section_name}]"
            if marker not in intro_text:
                issues["structure"].append(f"Missing section: {section_name}")
        
        # Check word count
        clean_text = self._clean_latex(intro_text)
        word_count = len(clean_text.split())
        if word_count < self.min_words:
            issues["content"].append(
                f"Too short ({word_count} words). Should be ≥{self.min_words} words."
            )
        
        # Check for citations
        sections = self._extract_sections(intro_text)
        for section_num, content in sections.items():
            if section_num in [1, 2, 3, 4]:  # These sections need citations
                if not re.search(r'\\citep?\{', content):
                    section_name = self.required_sections[section_num-1][0]
                    issues["citations"].append(
                        f"Section '{section_name}' needs citations"
                    )
        
        # Check for hypothesis/question
        if 5 in sections:
            hypothesis_section = sections[5]
            if not any(word in hypothesis_section.lower() for word in 
                      ["hypothesize", "ask", "question", "investigate", "explore"]):
                issues["content"].append(
                    "Research Question section should contain hypothesis or research question"
                )
        
        # Check flow between sections
        issues["flow"].extend(self._check_flow(sections))
        
        # Check for placeholder references
        if re.search(r'\\hlref\{', intro_text):
            count = len(re.findall(r'\\hlref\{([^}]+)\}', intro_text))
            issues["citations"].append(f"Found {count} placeholder references (\\hlref{{}})")
        
        return issues
    
    def _extract_sections(self, intro_text: str) -> Dict[int, str]:
        """Extract sections from introduction text."""
        sections = {}
        
        # Pattern to match sections
        pattern = r'\[START of (\d+)\. ([^\]]+)\](.*?)\[END of \d+\. [^\]]+\]'
        matches = re.finditer(pattern, intro_text, re.DOTALL)
        
        for match in matches:
            section_num = int(match.group(1))
            section_content = match.group(3).strip()
            sections[section_num] = section_content
        
        return sections
    
    def _check_flow(self, sections: Dict[int, str]) -> List[str]:
        """Check logical flow between sections."""
        flow_issues = []
        
        # Check if opening connects to importance
        if 1 in sections and 2 in sections:
            opening_concepts = self._extract_key_concepts(sections[1])
            importance_concepts = self._extract_key_concepts(sections[2])
            
            if not any(concept in importance_concepts for concept in opening_concepts):
                flow_issues.append(
                    "Opening statement concepts don't connect to Importance section"
                )
        
        # Check if limitations connect to research question
        if 4 in sections and 5 in sections:
            if "limitation" in sections[4].lower() and \
               not any(word in sections[5].lower() for word in 
                      ["overcome", "address", "solve", "tackle"]):
                flow_issues.append(
                    "Research question should address limitations mentioned"
                )
        
        return flow_issues
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts (nouns) from text."""
        # Simple approach: extract capitalized words and technical terms
        clean_text = self._clean_latex(text)
        words = clean_text.split()
        
        concepts = []
        for word in words:
            if len(word) > 4 and (word[0].isupper() or '-' in word):
                concepts.append(word.lower())
        
        return concepts
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands for text analysis."""
        # Remove section markers
        text = re.sub(r'\[START of.*?\]', '', text)
        text = re.sub(r'\[END of.*?\]', '', text)
        # Remove comments
        text = re.sub(r'%.*?\n', ' ', text)
        # Remove citations
        text = re.sub(r'\\citep?\{[^}]*\}', '', text)
        text = re.sub(r'\\hlref\{[^}]*\}', '', text)
        # Remove other commands
        text = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', ' ', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def suggest_transitions(self, sections: Dict[int, str]) -> Dict[str, str]:
        """Suggest transition phrases between sections."""
        transitions = {
            "1->2": "This [phenomenon/technology] has become increasingly important...",
            "2->3": "Extensive research over the past [timeframe] has revealed...",
            "3->4": "Despite these advances, several limitations persist...",
            "4->5": "To address these challenges, we hypothesized that...",
            "5->6": "Our approach leverages [key insight] to...",
            "6->7": "Through comprehensive experiments, we demonstrate...",
            "7->8": "These findings have important implications for..."
        }
        
        suggestions = {}
        for i in range(1, 8):
            if i in sections and i+1 in sections:
                key = f"{i}->{i+1}"
                if key in transitions:
                    suggestions[key] = transitions[key]
        
        return suggestions
    
    def check_paragraph_structure(self, intro_text: str) -> Dict[int, int]:
        """Check paragraph count in each section."""
        sections = self._extract_sections(intro_text)
        paragraph_counts = {}
        
        for section_num, content in sections.items():
            # Count paragraphs (double newlines)
            paragraphs = re.split(r'\n\s*\n', content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            paragraph_counts[section_num] = len(paragraphs)
        
        return paragraph_counts


def main():
    """Example usage."""
    agent = IntroductionAgent()
    
    # Create template
    project_info = {
        "field": "Neural signal processing",
        "problem": "phase-amplitude coupling analysis",
        "method": "GPU-accelerated computation"
    }
    
    template = agent.create_template(project_info)
    print("Generated Introduction Template")
    print("=" * 60)
    
    # Validate example
    issues = agent.validate(template)
    print("\nValidation Results:")
    for category, items in issues.items():
        if items:
            print(f"\n{category.title()}:")
            for item in items:
                print(f"  • {item}")


if __name__ == "__main__":
    main()