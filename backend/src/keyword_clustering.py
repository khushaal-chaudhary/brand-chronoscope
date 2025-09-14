"""
Keyword Clustering Module
Extracts and clusters keywords to discover themes - works well with small datasets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import logging
from pathlib import Path
import sys
import re

# Add parent path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordClusterer:
    """
    Discover topic clusters through keyword extraction and clustering
    Works better with small document sets than BERTopic
    """
    
    def __init__(self):
        self.vectorizer = None
        self.keywords_by_year = {}
        self.keyword_clusters = {}
        
    def extract_keywords_tfidf(self, df: pd.DataFrame, top_n: int = 30) -> Dict:
        """
        Extract important keywords using TF-IDF for each year
        
        Args:
            df: DataFrame with 'year' and 'text' columns
            top_n: Number of top keywords to extract per year
            
        Returns:
            Dictionary of keywords by year
        """
        keywords_by_year = {}
        
        for year in sorted(df['year'].unique()):
            year_docs = df[df['year'] == year]['text'].tolist()
            
            if not year_docs:
                continue
            
            # Combine all docs for the year
            year_text = ' '.join(str(doc) for doc in year_docs if doc)
            
            # Skip if text is too short
            if len(year_text) < 100:
                logger.warning(f"Text too short for year {year}")
                continue
            
            # TF-IDF with n-grams
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                ngram_range=(1, 3),  # Include bigrams and trigrams
                stop_words='english',
                min_df=1,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # At least 2 letters
            )
            
            try:
                # Fit on single combined document
                tfidf_matrix = vectorizer.fit_transform([year_text])
                
                # Get feature names (keywords)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get TF-IDF scores
                scores = tfidf_matrix.toarray()[0]
                
                # Create keyword-score pairs
                keywords = [(feature_names[i], scores[i]) 
                           for i in range(len(feature_names))]
                
                # Sort by score
                keywords.sort(key=lambda x: x[1], reverse=True)
                
                # Filter out very short or common terms
                filtered_keywords = []
                for keyword, score in keywords:
                    # Skip if too short or just numbers
                    if len(keyword) > 2 and not keyword.isdigit():
                        filtered_keywords.append((keyword, score))
                
                keywords_by_year[int(year)] = filtered_keywords[:top_n]
                
            except Exception as e:
                logger.warning(f"Error processing year {year}: {e}")
                continue
        
        self.keywords_by_year = keywords_by_year
        return keywords_by_year
    
    def identify_themes(self, keywords_by_year: Dict) -> Dict:
        """
        Group keywords into themes based on semantic similarity
        
        Args:
            keywords_by_year: Dictionary of keywords by year
            
        Returns:
            Dictionary mapping keywords to themes
        """
        # Collect all unique keywords
        all_keywords = set()
        for keywords in keywords_by_year.values():
            all_keywords.update([kw[0] for kw in keywords])
        
        # Manual theme mapping for tech/business terms
        # This is more reliable than clustering for small datasets
        theme_patterns = {
            'Technology & Innovation': [
                'innovation', 'technology', 'research', 'development', 'r&d',
                'artificial intelligence', 'ai', 'machine learning', 'ml',
                'silicon', 'chip', 'processor', 'neural', 'compute',
                'quantum', 'advanced', 'breakthrough', 'cutting edge'
            ],
            'AI & Intelligence': [
                'ai', 'artificial intelligence', 'machine learning', 'deep learning',
                'neural', 'copilot', 'chatgpt', 'gpt', 'openai', 'generative',
                'intelligence', 'cognitive', 'automation', 'bot'
            ],
            'Cloud & Platform': [
                'cloud', 'azure', 'aws', 'platform', 'infrastructure',
                'saas', 'paas', 'iaas', 'hybrid', 'edge computing',
                'serverless', 'container', 'kubernetes', 'microservices'
            ],
            'Privacy & Security': [
                'privacy', 'security', 'data protection', 'encryption',
                'confidential', 'secure', 'protection', 'safety',
                'cybersecurity', 'zero trust', 'compliance', 'gdpr'
            ],
            'Services & Ecosystem': [
                'services', 'ecosystem', 'platform', 'app store', 'marketplace',
                'subscription', 'recurring', 'saas', 'api', 'integration',
                'partners', 'developers', 'community'
            ],
            'Environmental & Social': [
                'environment', 'sustainability', 'carbon', 'renewable',
                'recycling', 'energy', 'climate', 'neutral', 'green',
                'social', 'diversity', 'inclusion', 'community', 'responsibility'
            ],
            'Financial Performance': [
                'revenue', 'growth', 'earnings', 'profit', 'margin',
                'sales', 'performance', 'results', 'quarter', 'fiscal',
                'investment', 'return', 'shareholder', 'value'
            ],
            'Products & Devices': [
                'iphone', 'ipad', 'mac', 'surface', 'xbox', 'hololens',
                'watch', 'airpods', 'vision pro', 'product', 'device', 
                'hardware', 'launch', 'release'
            ],
            'Digital Transformation': [
                'digital', 'transformation', 'modernization', 'digitization',
                'automation', 'workflow', 'productivity', 'efficiency',
                'remote', 'hybrid', 'collaboration'
            ],
            'Customer & Experience': [
                'customer', 'user', 'experience', 'satisfaction', 'engagement',
                'personalization', 'customization', 'support', 'success',
                'loyalty', 'retention', 'acquisition'
            ]
        }
        
        # Assign keywords to themes
        keyword_themes = {}
        
        for keyword in all_keywords:
            if not keyword:
                continue
                
            keyword_lower = keyword.lower()
            assigned = False
            
            # Check each theme
            for theme, patterns in theme_patterns.items():
                for pattern in patterns:
                    # Check if pattern is in keyword or keyword is in pattern
                    if pattern in keyword_lower or keyword_lower in pattern:
                        keyword_themes[keyword] = theme
                        assigned = True
                        break
                    # Also check for partial matches
                    pattern_words = pattern.split()
                    keyword_words = keyword_lower.split()
                    if any(pw in keyword_words for pw in pattern_words) or \
                       any(kw in pattern_words for kw in keyword_words):
                        keyword_themes[keyword] = theme
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                keyword_themes[keyword] = 'Other'
        
        return keyword_themes
    
    def analyze_theme_evolution(self, df: pd.DataFrame) -> Dict:
        """
        Complete analysis of theme evolution over time
        
        Args:
            df: DataFrame with 'year' and 'text' columns
            
        Returns:
            Dictionary with analysis results
        """
        # Extract keywords by year
        keywords_by_year = self.extract_keywords_tfidf(df, top_n=30)
        
        if not keywords_by_year:
            logger.error("No keywords extracted")
            return {}
        
        # Identify themes
        keyword_themes = self.identify_themes(keywords_by_year)
        
        # Track theme importance over time
        theme_evolution = {}
        
        for year, keywords in keywords_by_year.items():
            year_themes = defaultdict(float)
            
            for keyword, score in keywords:
                theme = keyword_themes.get(keyword, 'Other')
                year_themes[theme] += score
            
            # Normalize to percentages
            total = sum(year_themes.values())
            if total > 0:
                year_themes = {theme: (score/total)*100 
                              for theme, score in year_themes.items()}
            
            theme_evolution[year] = dict(year_themes)
        
        # Identify emerging themes
        emerging = self._find_emerging_themes(theme_evolution)
        
        # Identify declining themes
        declining = self._find_declining_themes(theme_evolution)
        
        return {
            'keywords_by_year': keywords_by_year,
            'themes': keyword_themes,
            'evolution': theme_evolution,
            'emerging': emerging,
            'declining': declining
        }
    
    def _find_emerging_themes(self, theme_evolution: Dict) -> List[Dict]:
        """
        Find themes that are growing in importance
        
        Args:
            theme_evolution: Dictionary of theme importance by year
            
        Returns:
            List of emerging theme dictionaries
        """
        years = sorted(theme_evolution.keys())
        
        if len(years) < 3:
            return []
        
        # Split into early and recent periods
        mid_point = len(years) // 2
        early_years = years[:mid_point]
        recent_years = years[mid_point:]
        
        # Get all themes
        all_themes = set()
        for year_themes in theme_evolution.values():
            all_themes.update(year_themes.keys())
        
        emerging = []
        
        for theme in all_themes:
            # Calculate average importance in each period
            early_values = [theme_evolution[y].get(theme, 0) for y in early_years]
            recent_values = [theme_evolution[y].get(theme, 0) for y in recent_years]
            
            early_avg = np.mean(early_values) if early_values else 0
            recent_avg = np.mean(recent_values) if recent_values else 0
            
            # Check if theme is emerging (at least 20% increase)
            if recent_avg > early_avg * 1.2 and recent_avg > 5:  # Must be significant
                growth = ((recent_avg - early_avg) / max(early_avg, 0.1)) * 100
                emerging.append({
                    'theme': theme,
                    'early_importance': round(early_avg, 1),
                    'recent_importance': round(recent_avg, 1),
                    'growth': round(growth, 0)
                })
        
        return sorted(emerging, key=lambda x: x['growth'], reverse=True)
    
    def _find_declining_themes(self, theme_evolution: Dict) -> List[Dict]:
        """
        Find themes that are declining in importance
        
        Args:
            theme_evolution: Dictionary of theme importance by year
            
        Returns:
            List of declining theme dictionaries
        """
        years = sorted(theme_evolution.keys())
        
        if len(years) < 3:
            return []
        
        mid_point = len(years) // 2
        early_years = years[:mid_point]
        recent_years = years[mid_point:]
        
        all_themes = set()
        for year_themes in theme_evolution.values():
            all_themes.update(year_themes.keys())
        
        declining = []
        
        for theme in all_themes:
            early_values = [theme_evolution[y].get(theme, 0) for y in early_years]
            recent_values = [theme_evolution[y].get(theme, 0) for y in recent_years]
            
            early_avg = np.mean(early_values) if early_values else 0
            recent_avg = np.mean(recent_values) if recent_values else 0
            
            # Check if theme is declining (at least 20% decrease)
            if recent_avg < early_avg * 0.8 and early_avg > 5:  # Was significant
                decline = ((recent_avg - early_avg) / max(early_avg, 0.1)) * 100
                declining.append({
                    'theme': theme,
                    'early_importance': round(early_avg, 1),
                    'recent_importance': round(recent_avg, 1),
                    'decline': round(decline, 0)
                })
        
        return sorted(declining, key=lambda x: x['decline'])
    
    def get_top_keywords_per_theme(self, keywords_by_year: Dict, 
                                   keyword_themes: Dict) -> Dict:
        """
        Get the top keywords for each theme across all years
        
        Args:
            keywords_by_year: Keywords extracted by year
            keyword_themes: Mapping of keywords to themes
            
        Returns:
            Dictionary of themes to top keywords
        """
        theme_keywords = defaultdict(list)
        
        # Aggregate all keywords with their scores
        for year, keywords in keywords_by_year.items():
            for keyword, score in keywords:
                theme = keyword_themes.get(keyword, 'Other')
                theme_keywords[theme].append((keyword, score))
        
        # Get top keywords per theme
        top_per_theme = {}
        for theme, kw_list in theme_keywords.items():
            # Sort by score and get unique keywords
            seen = set()
            unique_keywords = []
            for kw, score in sorted(kw_list, key=lambda x: x[1], reverse=True):
                if kw not in seen:
                    seen.add(kw)
                    unique_keywords.append(kw)
                if len(unique_keywords) >= 5:
                    break
            top_per_theme[theme] = unique_keywords
        
        return top_per_theme


# Test function
if __name__ == "__main__":
    print("Testing KeywordClusterer...")
    
    # Create test data
    test_df = pd.DataFrame({
        'year': [2020, 2021, 2022, 2023, 2024],
        'text': [
            'Cloud computing and artificial intelligence are transforming business',
            'AI and machine learning drive digital transformation',
            'Copilot and generative AI revolutionize productivity',
            'Security and privacy remain fundamental to our platform',
            'Quantum computing and AI will define the future'
        ]
    })
    
    clusterer = KeywordClusterer()
    results = clusterer.analyze_theme_evolution(test_df)
    
    print("\nKeywords by year:", results.get('keywords_by_year', {}))
    print("\nTheme evolution:", results.get('evolution', {}))
    print("\nEmerging themes:", results.get('emerging', []))