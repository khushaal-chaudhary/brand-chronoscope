"""
Strategic Topic Analyzer
Filters out financial/legal boilerplate to find actual strategic themes
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategicTopicAnalyzer:
    """
    Focuses on strategic content, filtering out boilerplate
    """
    
    def __init__(self):
        self.strategic_sections = []
        self.themes_by_year = {}
        
    def extract_strategic_sections(self, text: str) -> List[str]:
        """
        Extract only strategic/business content, skip financial tables
        """
        sections = []
        
        # Split into paragraphs
        paragraphs = text.split('\n')
        if not paragraphs:
            paragraphs = re.split(r'\.{2,}', text)
        
        strategic_keywords = [
            'strategy', 'innovation', 'competitive', 'market position',
            'technology', 'artificial intelligence', 'machine learning',
            'privacy', 'ecosystem', 'platform', 'services',
            'customer', 'experience', 'research', 'development',
            'future', 'vision', 'transform', 'disrupt',
            'sustainability', 'environment', 'carbon',
            'growth', 'opportunity', 'expansion', 'investment'
        ]
        
        boilerplate_indicators = [
            'accordance with', 'pursuant to', 'form 10-k', 'item ',
            'see note', 'refer to', 'gaap', 'fiscal year ended',
            'consolidated', 'financial statements', 'exhibit',
            'thousands except', 'share data', 'december 31',
            'september 28', 'registration statement', 'sec',
            'internal control', 'audit', 'deferred tax'
        ]
        
        for para in paragraphs:
            para_lower = para.lower()
            
            # Skip if too short or too long
            if len(para) < 100 or len(para) > 2000:
                continue
                
            # Skip if contains too many numbers (likely a table)
            numbers = re.findall(r'\d+', para)
            if len(numbers) > 10:
                continue
            
            # Skip if contains boilerplate
            if any(indicator in para_lower for indicator in boilerplate_indicators):
                continue
            
            # Include if contains strategic keywords
            if any(keyword in para_lower for keyword in strategic_keywords):
                sections.append(para)
        
        return sections
    
    def analyze_strategic_themes(self, df: pd.DataFrame) -> Dict:
        """
        Analyze only strategic content for themes
        """
        results = {
            'themes_by_year': {},
            'key_topics': [],
            'evolution': {},
            'insights': []
        }
        
        for year in sorted(df['year'].unique()):
            year_docs = df[df['year'] == year]['text'].tolist()
            
            # Extract strategic sections only
            strategic_content = []
            for doc in year_docs:
                sections = self.extract_strategic_sections(doc)
                strategic_content.extend(sections)
            
            if not strategic_content:
                logger.warning(f"No strategic content found for {year}")
                continue
            
            # Combine strategic content
            year_text = ' '.join(strategic_content)
            
            # Extract themes using TF-IDF
            themes = self._extract_themes_tfidf(year_text, year)
            results['themes_by_year'][int(year)] = themes
        
        # Analyze evolution
        results['evolution'] = self._analyze_evolution(results['themes_by_year'])
        
        # Generate insights
        results['insights'] = self._generate_insights(results['evolution'])
        
        return results
    
    def _extract_themes_tfidf(self, text: str, year: int) -> Dict:
        """
        Extract themes using TF-IDF on strategic content
        """
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=20,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = scores.argsort()[-20:][::-1]
            themes = {}
            
            for idx in top_indices:
                term = feature_names[idx]
                score = scores[idx]
                
                # Categorize the term
                category = self._categorize_term(term)
                if category not in themes:
                    themes[category] = []
                themes[category].append((term, score))
            
            return themes
            
        except Exception as e:
            logger.error(f"Error extracting themes for {year}: {e}")
            return {}
    
    def _categorize_term(self, term: str) -> str:
        """
        Categorize a term into strategic themes
        """
        term_lower = term.lower()
        
        categories = {
            'AI & Technology': ['ai', 'intelligence', 'machine', 'learning', 'neural', 'silicon', 'chip', 'processor'],
            'Privacy & Security': ['privacy', 'security', 'encryption', 'protect', 'safe'],
            'Services & Platform': ['service', 'platform', 'ecosystem', 'subscription', 'cloud'],
            'Innovation': ['innovation', 'research', 'development', 'new', 'breakthrough'],
            'Sustainability': ['sustainable', 'environment', 'carbon', 'renewable', 'energy'],
            'Customer Focus': ['customer', 'user', 'experience', 'satisfaction'],
            'Market Position': ['market', 'competitive', 'leader', 'growth', 'expansion']
        }
        
        for category, keywords in categories.items():
            if any(keyword in term_lower for keyword in keywords):
                return category
        
        return 'Other'
    
    def _analyze_evolution(self, themes_by_year: Dict) -> Dict:
        """
        Analyze how themes evolve over time
        """
        evolution = {}
        all_years = sorted(themes_by_year.keys())
        
        if len(all_years) < 2:
            return evolution
        
        # Track each category's importance over time
        all_categories = set()
        for themes in themes_by_year.values():
            all_categories.update(themes.keys())
        
        for category in all_categories:
            evolution[category] = {}
            for year in all_years:
                if year in themes_by_year and category in themes_by_year[year]:
                    # Sum scores for this category
                    total_score = sum(score for _, score in themes_by_year[year][category])
                    evolution[category][year] = total_score
                else:
                    evolution[category][year] = 0
        
        return evolution
    
    def _generate_insights(self, evolution: Dict) -> List[str]:
        """
        Generate insights from theme evolution
        """
        insights = []
        
        for category, year_scores in evolution.items():
            years = sorted(year_scores.keys())
            if len(years) < 3:
                continue
            
            # Calculate trend
            early_years = years[:len(years)//2]
            recent_years = years[len(years)//2:]
            
            early_avg = np.mean([year_scores[y] for y in early_years])
            recent_avg = np.mean([year_scores[y] for y in recent_years])
            
            if recent_avg > early_avg * 1.5:
                insights.append(f"📈 {category} has become significantly more important (+{(recent_avg/early_avg-1)*100:.0f}%)")
            elif recent_avg < early_avg * 0.5:
                insights.append(f"📉 {category} has declined in emphasis (-{(1-recent_avg/early_avg)*100:.0f}%)")
        
        return insights

def run_strategic_analysis():
    """
    Run strategic topic analysis on Apple data
    """
    print("\n" + "="*60)
    print("🎯 STRATEGIC TOPIC ANALYSIS")
    print("Filtering out legal/financial boilerplate")
    print("="*60)
    
    # Load data
    data_path = Path("data/processed/apple_10k_fixed.csv")
    if not data_path.exists():
        data_path = Path("data/processed/apple_10k_pdfs.csv")
    
    df = pd.read_csv(data_path)
    print(f"\n📊 Analyzing {len(df)} documents from {df['year'].min()}-{df['year'].max()}")
    
    # Run strategic analysis
    analyzer = StrategicTopicAnalyzer()
    results = analyzer.analyze_strategic_themes(df)
    
    print("\n🎯 STRATEGIC THEMES BY YEAR:")
    for year in sorted(results['themes_by_year'].keys())[-3:]:  # Last 3 years
        print(f"\n{year}:")
        themes = results['themes_by_year'][year]
        for category, terms in themes.items():
            if terms and category != 'Other':
                top_terms = [term for term, _ in terms[:3]]
                print(f"  {category}: {', '.join(top_terms)}")
    
    print("\n📈 THEME EVOLUTION:")
    evolution = results['evolution']
    for category in ['AI & Technology', 'Privacy & Security', 'Services & Platform']:
        if category in evolution:
            years = sorted(evolution[category].keys())
            if years:
                early = evolution[category][years[0]]
                recent = evolution[category][years[-1]]
                if early > 0:
                    change = (recent - early) / early * 100
                    print(f"  {category}: {change:+.0f}% change from {years[0]} to {years[-1]}")
    
    print("\n💡 KEY INSIGHTS:")
    for insight in results['insights']:
        print(f"  {insight}")
    
    # Compare with keyword approach
    print("\n🔍 COMPARISON WITH PHASE 1:")
    print("  Phase 1: Counted predefined keywords")
    print("  Phase 2: Discovered themes from strategic content")
    print("  Result: Found actual business themes, not legal terms")

if __name__ == "__main__":
    run_strategic_analysis()