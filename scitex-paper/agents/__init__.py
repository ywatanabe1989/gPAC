"""
SciTeX Agents: Specialized handlers for scientific manuscript sections
"""

from .base import BaseAgent
from .abstract_agent import AbstractAgent
from .introduction_agent import IntroductionAgent
from .literature_agent import LiteratureAgent

__all__ = [
    'BaseAgent',
    'AbstractAgent', 
    'IntroductionAgent',
    'LiteratureAgent',
]

# Agent registry for easy access
AGENTS = {
    'abstract': AbstractAgent,
    'introduction': IntroductionAgent,
    'literature': LiteratureAgent,
}

def get_agent(section_type: str):
    """Get appropriate agent for a section type."""
    agent_class = AGENTS.get(section_type)
    if agent_class:
        return agent_class()
    else:
        raise ValueError(f"No agent available for section: {section_type}")