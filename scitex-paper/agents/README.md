# SciTeX Specialized Agents

Modular agents for handling specific aspects of scientific manuscript writing. Each agent focuses on a particular section or task, ensuring best practices and consistency.

## Available Agents

### 1. Abstract Agent (`abstract_agent.py`)
Specializes in abstract writing with strict 7-section structure.

**Features:**
- Validates 150-250 word limit
- Ensures 7-section structure
- Checks for "Here we show..." statement
- Validates presence of comparisons
- Scores abstract quality (0-100)

**Usage:**
```python
from agents.abstract_agent import AbstractAgent

agent = AbstractAgent()

# Create template
template = agent.create_template({
    'background_hint': 'Phase-amplitude coupling',
    'method_hint': 'GPU acceleration'
})

# Validate existing abstract
issues = agent.validate(abstract_text)

# Get improvement suggestions
suggestions = agent.suggest_improvements(abstract_text)

# Score abstract
scores = agent.score_abstract(abstract_text)  # Returns dict with scores
```

### 2. Introduction Agent (`introduction_agent.py`)
Manages introduction with 8-section structure and logical flow.

**Features:**
- Validates ≥1000 word requirement
- Ensures 8 required sections
- Checks citation presence
- Validates logical flow between sections
- Suggests transition phrases

**Usage:**
```python
from agents.introduction_agent import IntroductionAgent

agent = IntroductionAgent()

# Create structured template
template = agent.create_template({
    'field': 'Neural signal processing',
    'problem': 'computational bottlenecks',
    'method': 'GPU parallelization'
})

# Check paragraph structure
para_counts = agent.check_paragraph_structure(intro_text)

# Get transition suggestions
transitions = agent.suggest_transitions(sections)
```

### 3. Literature Agent (`literature_agent.py`)
Automates literature search and reference management.

**Features:**
- Multi-database search (Semantic Scholar, arXiv, PubMed)
- Semantic paper similarity
- Citation network exploration
- Reference quality analysis
- BibTeX generation

**Usage:**
```python
from agents.literature_agent import LiteratureAgent

agent = LiteratureAgent()

# Search papers
papers = agent.search_papers(
    "phase amplitude coupling GPU",
    sources=['semantic_scholar', 'arxiv'],
    year_range=(2020, 2024)
)

# Find related papers
related = agent.find_related_papers(paper_id, method='citations')

# Generate BibTeX
bibtex = agent.create_bibliography_entry(paper, style='bibtex')

# Analyze reference quality
quality = agent.check_reference_quality(current_references)

# Create reading list
reading_list = agent.create_reading_list("neural oscillations")
```

### 4. Methods Agent (Coming Soon)
- Passive voice enforcement
- Reproducibility checklist
- Statistical methods validation
- Code/data availability statements

### 5. Results Agent (Coming Soon)
- Statistical reporting validation
- Figure/table reference checking
- Quantitative claim verification
- Effect size reporting

### 6. Discussion Agent (Coming Soon)
- 5-section structure validation
- Limitation acknowledgment
- Future work suggestions
- Implication clarity

### 7. Figure Agent (Coming Soon)
- Axis label validation
- Unit checking
- Color accessibility
- Resolution verification

### 8. Statistics Agent (Coming Soon)
- Complete reporting validation
- Multiple comparison corrections
- Assumption checking
- Power analysis

## Integration with SciTeX

### Automatic Validation
```python
# In scitex.py
from agents import AbstractAgent, IntroductionAgent

def validate_manuscript(self):
    """Run all agent validations."""
    agents = {
        'abstract': AbstractAgent(),
        'introduction': IntroductionAgent(),
        # ... other agents
    }
    
    all_issues = {}
    for section, agent in agents.items():
        section_file = self.get_section_file(section)
        if section_file.exists():
            issues = agent.validate(section_file.read_text())
            if any(issues.values()):
                all_issues[section] = issues
    
    return all_issues
```

### AI-Assisted Writing
```python
# Generate section with guidance
def create_guided_section(self, section_type: str):
    """Create section with agent guidance."""
    agent = self.get_agent(section_type)
    
    # Get project context
    context = self.extract_project_context()
    
    # Generate template
    template = agent.create_template(context)
    
    # Add validation hints
    template += f"\n\n% Validation Criteria:\n"
    for criterion in agent.get_criteria():
        template += f"% - {criterion}\n"
    
    return template
```

## Best Practices

1. **Run validation frequently** - Catch issues early
2. **Use templates** - Start with proper structure
3. **Check references** - Ensure citations are complete
4. **Monitor metrics** - Track word counts, statistics
5. **Iterate improvements** - Use agent suggestions

## Extending Agents

Create custom agents by inheriting base functionality:

```python
from agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.criteria = [...]
    
    def validate(self, text: str) -> Dict:
        # Your validation logic
        pass
    
    def create_template(self, context: Dict) -> str:
        # Your template generation
        pass
```

## Requirements

```bash
# Core functionality
pip install pyyaml

# Literature search
pip install semanticscholar
pip install requests

# Advanced NLP (optional)
pip install spacy
pip install transformers
```

## Roadmap

- [ ] Real-time validation in editors
- [ ] LLM-powered suggestions
- [ ] Journal style adaptation
- [ ] Collaborative editing support
- [ ] Citation context analysis
- [ ] Automated literature reviews