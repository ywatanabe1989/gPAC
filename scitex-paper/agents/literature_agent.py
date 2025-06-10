#!/usr/bin/env python3
"""
Literature Agent: Automated literature search and reference management
Integrates with scientific databases for semantic search.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

# Optional imports for various APIs
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from semanticscholar import SemanticScholar
    S2_AVAILABLE = True
except ImportError:
    S2_AVAILABLE = False


class LiteratureAgent:
    """Agent for automated literature search and management."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.scitex' / 'literature_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints (free/open access)
        self.apis = {
            'semantic_scholar': 'https://api.semanticscholar.org/v1/paper/search',
            'arxiv': 'http://export.arxiv.org/api/query',
            'pubmed': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
            'crossref': 'https://api.crossref.org/works',
            # 'openalex': 'https://api.openalex.org/works',  # New free API
        }
        
        # Initialize S2 client if available
        self.s2_client = SemanticScholar() if S2_AVAILABLE else None
        
    def search_papers(self, query: str, 
                     sources: List[str] = ['semantic_scholar'],
                     max_results: int = 20,
                     year_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """Search for papers across multiple databases."""
        all_results = []
        
        for source in sources:
            if source == 'semantic_scholar' and S2_AVAILABLE:
                results = self._search_semantic_scholar(query, max_results, year_range)
            elif source == 'arxiv':
                results = self._search_arxiv(query, max_results)
            elif source == 'pubmed':
                results = self._search_pubmed(query, max_results)
            else:
                continue
                
            all_results.extend(results)
        
        # Remove duplicates based on title similarity
        unique_results = self._deduplicate_results(all_results)
        
        return unique_results
    
    def _search_semantic_scholar(self, query: str, 
                                max_results: int,
                                year_range: Optional[Tuple[int, int]]) -> List[Dict]:
        """Search Semantic Scholar using their API."""
        if not self.s2_client:
            return []
        
        try:
            # Build query with year filter
            search_query = query
            if year_range:
                search_query += f" year:{year_range[0]}-{year_range[1]}"
            
            results = self.s2_client.search_paper(
                search_query,
                limit=max_results,
                fields=['title', 'abstract', 'authors', 'year', 'venue', 
                       'citationCount', 'doi', 'url']
            )
            
            papers = []
            for paper in results:
                papers.append({
                    'title': paper.title,
                    'authors': [a.name for a in paper.authors],
                    'year': paper.year,
                    'abstract': paper.abstract,
                    'venue': paper.venue,
                    'citations': paper.citationCount,
                    'doi': paper.doi,
                    'url': paper.url,
                    'source': 'semantic_scholar',
                    'relevance_score': 0  # Will be calculated
                })
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []
    
    def _search_arxiv(self, query: str, max_results: int) -> List[Dict]:
        """Search arXiv using their API."""
        if not REQUESTS_AVAILABLE:
            return []
        
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.apis['arxiv'], params=params)
            
            # Parse XML response (simplified)
            papers = []
            # Would need proper XML parsing here
            
            return papers
            
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []
    
    def _search_pubmed(self, query: str, max_results: int) -> List[Dict]:
        """Search PubMed using E-utilities."""
        # Implementation would go here
        return []
    
    def find_related_papers(self, paper_id: str, 
                           method: str = 'citations') -> List[Dict]:
        """Find papers related to a given paper."""
        if method == 'citations':
            return self._get_citations(paper_id)
        elif method == 'references':
            return self._get_references(paper_id)
        elif method == 'similar':
            return self._get_similar_papers(paper_id)
        else:
            return []
    
    def _get_citations(self, paper_id: str) -> List[Dict]:
        """Get papers that cite this paper."""
        if not self.s2_client:
            return []
        
        try:
            paper = self.s2_client.get_paper(paper_id)
            citations = paper.citations
            
            return [self._paper_to_dict(c) for c in citations[:20]]
        except:
            return []
    
    def _get_references(self, paper_id: str) -> List[Dict]:
        """Get papers referenced by this paper."""
        if not self.s2_client:
            return []
        
        try:
            paper = self.s2_client.get_paper(paper_id)
            references = paper.references
            
            return [self._paper_to_dict(r) for r in references[:20]]
        except:
            return []
    
    def generate_search_queries(self, 
                               topic: str,
                               paper_type: str = 'research') -> List[str]:
        """Generate effective search queries for a topic."""
        base_queries = []
        
        if paper_type == 'research':
            templates = [
                f'"{topic}"',
                f'{topic} AND (method OR algorithm OR approach)',
                f'{topic} AND (evaluation OR comparison OR benchmark)',
                f'{topic} AND (state-of-the-art OR SOTA)',
                f'{topic} AND review',
            ]
        elif paper_type == 'application':
            templates = [
                f'{topic} AND application',
                f'{topic} AND (clinical OR practical)',
                f'{topic} AND "case study"',
                f'{topic} AND implementation',
            ]
        elif paper_type == 'theory':
            templates = [
                f'{topic} AND (theory OR theoretical)',
                f'{topic} AND (proof OR theorem)',
                f'{topic} AND mathematical',
                f'{topic} AND foundations',
            ]
        else:
            templates = [f'"{topic}"']
        
        return templates
    
    def extract_key_papers(self, 
                          papers: List[Dict],
                          criteria: str = 'citations') -> List[Dict]:
        """Extract most important papers based on criteria."""
        if criteria == 'citations':
            # Sort by citation count
            sorted_papers = sorted(papers, 
                                 key=lambda p: p.get('citations', 0), 
                                 reverse=True)
        elif criteria == 'recency':
            # Sort by year
            sorted_papers = sorted(papers,
                                 key=lambda p: p.get('year', 0),
                                 reverse=True)
        elif criteria == 'relevance':
            # Sort by relevance score
            sorted_papers = sorted(papers,
                                 key=lambda p: p.get('relevance_score', 0),
                                 reverse=True)
        else:
            sorted_papers = papers
        
        return sorted_papers[:10]  # Top 10
    
    def create_bibliography_entry(self, paper: Dict, style: str = 'bibtex') -> str:
        """Create bibliography entry in specified style."""
        if style == 'bibtex':
            # Generate BibTeX entry
            authors = ' and '.join(paper.get('authors', ['Unknown']))
            year = paper.get('year', 'n.d.')
            title = paper.get('title', 'Untitled')
            venue = paper.get('venue', '')
            
            # Create citation key
            first_author = paper.get('authors', ['Unknown'])[0].split()[-1]
            cite_key = f"{first_author}{year}"
            
            entry = f"""@article{{{cite_key},
    title = {{{title}}},
    author = {{{authors}}},
    year = {{{year}}},
    journal = {{{venue}}},
"""
            
            if paper.get('doi'):
                entry += f"    doi = {{{paper['doi']}}},\n"
            if paper.get('url'):
                entry += f"    url = {{{paper['url']}}},\n"
                
            entry += "}"
            
            return entry
            
        return ""
    
    def suggest_missing_references(self, 
                                  current_refs: List[str],
                                  topic: str) -> List[Dict]:
        """Suggest important papers that might be missing."""
        # Search for seminal papers
        queries = [
            f'{topic} seminal',
            f'{topic} "highly cited"',
            f'{topic} review',
            f'{topic} survey',
        ]
        
        suggestions = []
        for query in queries:
            results = self.search_papers(query, max_results=10)
            suggestions.extend(results)
        
        # Filter out papers already cited
        # (This would need actual implementation to match titles)
        
        # Extract most cited papers
        key_papers = self.extract_key_papers(suggestions, criteria='citations')
        
        return key_papers
    
    def check_reference_quality(self, references: List[Dict]) -> Dict[str, any]:
        """Analyze the quality of a reference list."""
        analysis = {
            'total_refs': len(references),
            'year_distribution': {},
            'venue_distribution': {},
            'citation_stats': {
                'mean': 0,
                'median': 0,
                'highly_cited': 0
            },
            'issues': []
        }
        
        # Analyze year distribution
        years = [r.get('year', 0) for r in references if r.get('year')]
        if years:
            current_year = datetime.now().year
            recent_refs = sum(1 for y in years if current_year - y <= 5)
            
            if recent_refs / len(years) < 0.3:
                analysis['issues'].append(
                    "Less than 30% of references are from the last 5 years"
                )
        
        # Analyze citation impact
        citations = [r.get('citations', 0) for r in references]
        if citations:
            analysis['citation_stats']['mean'] = sum(citations) / len(citations)
            analysis['citation_stats']['highly_cited'] = sum(1 for c in citations if c > 100)
        
        return analysis
    
    def _deduplicate_results(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity."""
        unique = []
        seen_titles = set()
        
        for paper in papers:
            # Simple deduplication by normalized title
            normalized = re.sub(r'[^a-z0-9]', '', paper['title'].lower())
            
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique.append(paper)
        
        return unique
    
    def _paper_to_dict(self, paper) -> Dict:
        """Convert paper object to dictionary."""
        return {
            'title': getattr(paper, 'title', ''),
            'authors': [a.name for a in getattr(paper, 'authors', [])],
            'year': getattr(paper, 'year', None),
            'venue': getattr(paper, 'venue', ''),
            'citations': getattr(paper, 'citationCount', 0),
            'doi': getattr(paper, 'doi', ''),
            'source': 'semantic_scholar'
        }
    
    def create_reading_list(self, topic: str, max_papers: int = 20) -> str:
        """Create a curated reading list for a topic."""
        reading_list = f"# Reading List: {topic}\n\n"
        reading_list += f"Generated: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # Get seminal papers
        reading_list += "## Seminal Papers\n"
        seminal = self.search_papers(f"{topic} seminal", max_results=5)
        for i, paper in enumerate(seminal[:5], 1):
            reading_list += f"{i}. **{paper['title']}** ({paper.get('year', 'n.d.')})\n"
            reading_list += f"   - Authors: {', '.join(paper.get('authors', [])[:3])}\n"
            reading_list += f"   - Citations: {paper.get('citations', 0)}\n\n"
        
        # Get recent papers
        reading_list += "\n## Recent Advances\n"
        recent = self.search_papers(topic, max_results=10, 
                                   year_range=(datetime.now().year-2, datetime.now().year))
        for i, paper in enumerate(recent[:5], 1):
            reading_list += f"{i}. **{paper['title']}** ({paper.get('year', 'n.d.')})\n"
            reading_list += f"   - Venue: {paper.get('venue', 'Unknown')}\n\n"
        
        return reading_list


def main():
    """Example usage."""
    agent = LiteratureAgent()
    
    # Example search
    topic = "phase amplitude coupling GPU"
    print(f"Searching for papers on: {topic}")
    print("=" * 60)
    
    # Generate search queries
    queries = agent.generate_search_queries(topic, paper_type='research')
    print("\nGenerated queries:")
    for q in queries:
        print(f"  • {q}")
    
    # Create reading list
    reading_list = agent.create_reading_list(topic)
    print("\n" + reading_list)
    
    # Note about API requirements
    print("\nNote: Full functionality requires:")
    print("  • pip install semanticscholar")
    print("  • pip install requests")
    print("  • API keys for some services")


if __name__ == "__main__":
    main()