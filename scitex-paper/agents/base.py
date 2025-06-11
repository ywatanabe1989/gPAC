#!/usr/bin/env python3
"""
Base Agent: Abstract base class for all specialized agents
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any


class BaseAgent(ABC):
    """Abstract base class for specialized manuscript agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.validation_cache = {}
        
    @abstractmethod
    def validate(self, text: str) -> Dict[str, List[str]]:
        """Validate text according to agent's rules.
        
        Returns:
            Dict with categories of issues found
        """
        pass
    
    @abstractmethod
    def create_template(self, context: Dict[str, Any]) -> str:
        """Create section template with context.
        
        Args:
            context: Project-specific information
            
        Returns:
            LaTeX template string
        """
        pass
    
    def suggest_improvements(self, text: str) -> List[str]:
        """Suggest specific improvements.
        
        Default implementation based on validation results.
        """
        issues = self.validate(text)
        suggestions = []
        
        for category, items in issues.items():
            for item in items:
                suggestion = self._issue_to_suggestion(item, category)
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _issue_to_suggestion(self, issue: str, category: str) -> Optional[str]:
        """Convert validation issue to actionable suggestion."""
        # Default mappings
        if "too short" in issue.lower():
            return "Expand this section with more detail"
        elif "too long" in issue.lower():
            return "Condense this section to meet word limits"
        elif "missing" in issue.lower():
            return f"Add the {issue.split('missing')[-1].strip()}"
        else:
            return f"Fix {category}: {issue}"
    
    def score(self, text: str) -> Dict[str, float]:
        """Score text quality (0-100).
        
        Default scoring based on validation issues.
        """
        issues = self.validate(text)
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points for issues
        for category, items in issues.items():
            if category == 'critical':
                score -= len(items) * 20
            elif category == 'major':
                score -= len(items) * 10
            else:
                score -= len(items) * 5
        
        return {
            'overall': max(0, score),
            'categories': {cat: 100 - len(items)*10 for cat, items in issues.items()}
        }
    
    def get_criteria(self) -> List[str]:
        """Get validation criteria for this agent."""
        return getattr(self, 'criteria', [])
    
    def get_guidelines(self) -> str:
        """Get writing guidelines for this section."""
        return getattr(self, 'guidelines', "Follow best practices")
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text (word count, etc)."""
        clean_text = self._clean_latex(text)
        
        return {
            'word_count': len(clean_text.split()),
            'sentence_count': len(clean_text.split('.')),
            'paragraph_count': len(clean_text.split('\n\n')),
        }
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands for text analysis.
        
        Override in subclasses for specific cleaning.
        """
        import re
        
        # Remove comments
        text = re.sub(r'%.*?\n', ' ', text)
        # Remove commands
        text = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', ' ', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()