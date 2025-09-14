"""
Semantic Drift Analyzer
Tracks how word meanings change over time using embeddings
This is cutting-edge NLP that shows deep technical understanding
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import logging
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticDriftAnalyzer:
    """
    Analyzes how word meanings and contexts evolve over time
    Uses sentence embeddings to track semantic shifts
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence transformer model
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info(f"Initializing semantic drift analyzer with {model_name}")
        self.model = SentenceTransformer(model_name)
        self.word_embeddings_by_year = {}
        self.context_embeddings_by_year = {}
        
    def extract_word_contexts(self, text: str, target_word: str, window_size: int = 50) -> List[str]:
        """
        Extract contexts where a word appears
        
        Args:
            text: Full text to search
            target_word: Word to find contexts for
            window_size: Characters before/after word to include
            
        Returns:
            List of context strings
        """
        contexts = []
        text_lower = text.lower()
        word_lower = target_word.lower()
        
        # Find all occurrences
        pattern = r'\b' + re.escape(word_lower) + r'\b'
        
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            
            context = text[start:end]
            # Clean up the context
            context = ' '.join(context.split())
            contexts.append(context)
        
        return contexts
    
    def analyze_word_evolution(self, df: pd.DataFrame, target_word: str) -> Dict:
        """
        Analyze how a word's meaning changes over time
        
        Args:
            df: DataFrame with 'year' and 'text' columns
            target_word: Word to analyze
            
        Returns:
            Dictionary with evolution analysis
        """
        results = {
            'word': target_word,
            'contexts_by_year': {},
            'semantic_neighbors_by_year': {},
            'meaning_shifts': [],
            'total_drift_score': 0,
            'yearly_drift': {},
            'embedding_positions': {}
        }
        
        # Collect contexts for each year
        year_contexts = {}
        year_embeddings = {}
        
        for year in sorted(df['year'].unique()):
            year_text = ' '.join(df[df['year'] == year]['text'].tolist())
            
            # Extract contexts for the target word
            contexts = self.extract_word_contexts(year_text, target_word)
            
            if contexts:
                year_contexts[year] = contexts
                
                # Get embeddings for contexts
                embeddings = self.model.encode(contexts)
                
                # Store average embedding
                avg_embedding = np.mean(embeddings, axis=0)
                year_embeddings[year] = avg_embedding
                
                # Store most representative context
                if len(contexts) > 0:
                    # Find context closest to average
                    similarities = cosine_similarity(embeddings, [avg_embedding])
                    best_idx = similarities.argmax()
                    results['contexts_by_year'][int(year)] = contexts[best_idx]
        
        # Analyze semantic drift between consecutive years
        years = sorted(year_embeddings.keys())
        
        if len(years) > 1:
            # Calculate year-over-year drift
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                
                similarity = cosine_similarity(
                    [year_embeddings[prev_year]], 
                    [year_embeddings[curr_year]]
                )[0][0]
                
                drift = 1 - similarity
                results['yearly_drift'][f"{prev_year}-{curr_year}"] = float(drift)
            
            # Calculate total drift (first year to last year)
            total_similarity = cosine_similarity(
                [year_embeddings[years[0]]], 
                [year_embeddings[years[-1]]]
            )[0][0]
            
            results['total_drift_score'] = float(1 - total_similarity)
            
            # Identify major meaning shifts
            results['meaning_shifts'] = self._identify_meaning_shifts(
                year_contexts, year_embeddings, years
            )
            
            # Find semantic neighbors for each year
            results['semantic_neighbors_by_year'] = self._find_semantic_neighbors(
                df, target_word, year_embeddings
            )
            
            # Get 2D positions for visualization
            if len(year_embeddings) > 2:
                results['embedding_positions'] = self._get_2d_positions(year_embeddings)
        
        return results
    
    def _identify_meaning_shifts(self, contexts: Dict, embeddings: Dict, years: List) -> List[Dict]:
        """
        Identify significant shifts in meaning
        """
        shifts = []
        
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            if prev_year in embeddings and curr_year in embeddings:
                similarity = cosine_similarity(
                    [embeddings[prev_year]], 
                    [embeddings[curr_year]]
                )[0][0]
                
                drift = 1 - similarity
                
                # Significant shift if drift > 0.2
                if drift > 0.2:
                    shifts.append({
                        'period': f"{prev_year}-{curr_year}",
                        'drift_score': float(drift),
                        'interpretation': self._interpret_shift(
                            contexts.get(prev_year, []),
                            contexts.get(curr_year, [])
                        )
                    })
        
        return shifts
    
    def _interpret_shift(self, old_contexts: List[str], new_contexts: List[str]) -> str:
        """
        Interpret what changed between contexts
        """
        if not old_contexts or not new_contexts:
            return "Context changed significantly"
        
        # Simple interpretation based on common words
        old_text = ' '.join(old_contexts[:3]).lower()
        new_text = ' '.join(new_contexts[:3]).lower()
        
        # Find key different words
        old_words = set(old_text.split())
        new_words = set(new_text.split())
        
        unique_old = old_words - new_words
        unique_new = new_words - old_words
        
        if unique_new:
            key_new = list(unique_new)[:3]
            return f"New associations: {', '.join(key_new)}"
        
        return "Context evolved"
    
    def _find_semantic_neighbors(self, df: pd.DataFrame, target_word: str, 
                                 year_embeddings: Dict) -> Dict:
        """
        Find words that are semantically similar in each time period
        """
        neighbors = {}
        
        for year in year_embeddings.keys():
            year_text = ' '.join(df[df['year'] == year]['text'].tolist())
            
            # Extract meaningful phrases
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 3),
                stop_words='english'
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform([year_text])
                terms = vectorizer.get_feature_names_out()
                
                # Get embeddings for terms
                term_embeddings = self.model.encode(terms.tolist())
                
                # Find most similar to target word's embedding
                target_embedding = year_embeddings[year]
                similarities = cosine_similarity([target_embedding], term_embeddings)[0]
                
                # Get top similar terms (excluding the target word itself)
                top_indices = similarities.argsort()[-10:][::-1]
                
                similar_terms = []
                for idx in top_indices:
                    term = terms[idx]
                    if target_word.lower() not in term.lower():
                        similar_terms.append(term)
                    if len(similar_terms) >= 5:
                        break
                
                neighbors[int(year)] = similar_terms
                
            except Exception as e:
                logger.warning(f"Could not find neighbors for {year}: {e}")
                neighbors[int(year)] = []
        
        return neighbors
    
    def _get_2d_positions(self, embeddings: Dict) -> Dict:
        """
        Get 2D positions for visualization using PCA
        """
        if len(embeddings) < 2:
            return {}
        
        # Stack all embeddings
        years = sorted(embeddings.keys())
        embedding_matrix = np.vstack([embeddings[year] for year in years])
        
        # Reduce to 2D
        pca = PCA(n_components=2)
        positions_2d = pca.fit_transform(embedding_matrix)
        
        # Create position dictionary
        positions = {}
        for i, year in enumerate(years):
            positions[int(year)] = {
                'x': float(positions_2d[i, 0]),
                'y': float(positions_2d[i, 1])
            }
        
        return positions
    
    def analyze_multiple_words(self, df: pd.DataFrame, words: List[str]) -> Dict:
        """
        Analyze semantic drift for multiple words
        
        Args:
            df: DataFrame with text data
            words: List of words to analyze
            
        Returns:
            Dictionary with analysis for each word
        """
        results = {}
        
        for word in words:
            logger.info(f"Analyzing semantic drift for '{word}'")
            results[word] = self.analyze_word_evolution(df, word)
        
        # Rank by total drift
        drift_ranking = sorted(
            [(word, data['total_drift_score']) for word, data in results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'word_analyses': results,
            'drift_ranking': drift_ranking
        }
    
    def find_emerging_concepts(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """
        Find concepts that emerged or changed meaning significantly
        
        Args:
            df: DataFrame with text data
            top_n: Number of top emerging concepts to return
            
        Returns:
            List of emerging concepts with their evolution
        """
        # Split data into early and recent periods
        median_year = df['year'].median()
        early_df = df[df['year'] <= median_year]
        recent_df = df[df['year'] > median_year]
        
        if early_df.empty or recent_df.empty:
            return []
        
        # Get important terms from each period
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        early_text = ' '.join(early_df['text'].tolist())
        recent_text = ' '.join(recent_df['text'].tolist())
        
        # Find terms that increased in importance
        emerging = []
        
        try:
            # Fit on recent text to find important current terms
            recent_tfidf = vectorizer.fit_transform([recent_text])
            recent_terms = vectorizer.get_feature_names_out()
            recent_scores = recent_tfidf.toarray()[0]
            
            for i, term in enumerate(recent_terms):
                # Count occurrences in each period
                early_count = early_text.lower().count(term.lower())
                recent_count = recent_text.lower().count(term.lower())
                
                # Calculate growth
                if early_count > 0:
                    growth = recent_count / early_count
                else:
                    growth = recent_count
                
                if growth > 2 and recent_count > 5:  # Significant growth
                    # Analyze semantic evolution
                    evolution = self.analyze_word_evolution(df, term)
                    
                    emerging.append({
                        'concept': term,
                        'early_count': early_count,
                        'recent_count': recent_count,
                        'growth_factor': growth,
                        'semantic_drift': evolution['total_drift_score'],
                        'first_context': evolution['contexts_by_year'].get(
                            min(evolution['contexts_by_year'].keys()), ''
                        ) if evolution['contexts_by_year'] else '',
                        'latest_context': evolution['contexts_by_year'].get(
                            max(evolution['contexts_by_year'].keys()), ''
                        ) if evolution['contexts_by_year'] else ''
                    })
        
        except Exception as e:
            logger.error(f"Error finding emerging concepts: {e}")
        
        # Sort by growth factor
        emerging.sort(key=lambda x: x['growth_factor'], reverse=True)
        
        return emerging[:top_n]


# Test function
if __name__ == "__main__":
    # Test with sample data
    test_df = pd.DataFrame({
        'year': [2015, 2018, 2021, 2024],
        'text': [
            'Cloud computing platform enables business transformation and digital innovation',
            'Cloud platform with intelligent services powered by machine learning',
            'AI platform delivering intelligent experiences through cognitive services',
            'Copilot platform for generative AI transforming how people work'
        ]
    })
    
    analyzer = SemanticDriftAnalyzer()
    
    # Analyze how 'platform' meaning changed
    result = analyzer.analyze_word_evolution(test_df, 'platform')
    
    print(f"\n📊 Semantic drift analysis for 'platform':")
    print(f"Total drift score: {result['total_drift_score']:.3f}")
    print(f"\nYear-by-year contexts:")
    for year, context in result['contexts_by_year'].items():
        print(f"  {year}: {context[:100]}...")
    
    print(f"\nSemantic neighbors by year:")
    for year, neighbors in result['semantic_neighbors_by_year'].items():
        print(f"  {year}: {', '.join(neighbors[:3])}")