"""
Keyword Analysis Module for Brand Chronoscope
Core NLP logic for tracking brand language evolution
FIXED VERSION - Resolves KeyError issues
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """Analyzes keyword trends in corporate communications"""
    
    def __init__(self):
        """Initialize with predefined marketing keywords"""
        
        # Default keywords for tracking (can be customized)
        self.default_keywords = {
            # Technology & Innovation
            'innovation': ['innovation', 'innovative', 'innovate'],
            'ai': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural'],
            'cloud': ['cloud', 'saas', 'serverless', 'infrastructure'],
            'data': ['data', 'analytics', 'insights', 'metrics'],
            
            # Brand Values
            'privacy': ['privacy', 'private', 'confidential', 'secure'],
            'sustainability': ['sustainable', 'sustainability', 'environment', 'carbon', 'renewable'],
            'customer': ['customer', 'user', 'experience', 'satisfaction'],
            'quality': ['quality', 'premium', 'excellence', 'superior'],
            
            # Digital Transformation
            'digital': ['digital', 'digitization', 'digitalization', 'online'],
            'transformation': ['transformation', 'transform', 'change', 'evolve'],
            'platform': ['platform', 'ecosystem', 'integration'],
            'automation': ['automation', 'automate', 'automated', 'efficiency'],
            
            # Emerging Tech
            'blockchain': ['blockchain', 'crypto', 'decentralized'],
            'metaverse': ['metaverse', 'virtual', 'augmented', 'vr', 'ar'],
            '5g': ['5g', 'connectivity', 'network', 'wireless'],
            'quantum': ['quantum', 'computing']
        }
        
        self.tracked_keywords = []
        self.analysis_results = None
    
    def set_keywords(self, keywords: List[str] = None, use_categories: bool = False):
        """
        Set keywords to track
        
        Args:
            keywords: Custom list of keywords
            use_categories: Whether to use category-based keywords
        """
        if keywords:
            self.tracked_keywords = [k.lower() for k in keywords]
        elif use_categories:
            # Flatten all category keywords
            self.tracked_keywords = list(set([
                word.lower() 
                for category in self.default_keywords.values() 
                for word in category
            ]))
        else:
            # Use main category names
            self.tracked_keywords = list(self.default_keywords.keys())
        
        logger.info(f"Tracking {len(self.tracked_keywords)} keywords: {self.tracked_keywords[:5]}...")
        return self.tracked_keywords
    
    def analyze_keyword_trends(self, df: pd.DataFrame, 
                              keywords: List[str] = None,
                              normalize: bool = True) -> pd.DataFrame:
        """
        Analyze keyword frequency trends over time
        
        Args:
            df: Processed DataFrame with 'year', 'tokens' columns
            keywords: Keywords to analyze (uses default if None)
            normalize: Whether to normalize frequencies
        
        Returns:
            DataFrame with keyword frequencies by year
        """
        # Validate input
        if df is None or df.empty:
            logger.error("Input DataFrame is empty or None")
            return pd.DataFrame()
        
        if 'year' not in df.columns or 'tokens' not in df.columns:
            logger.error(f"Required columns missing. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Set keywords
        if keywords:
            self.set_keywords(keywords)
        elif not self.tracked_keywords:
            self.set_keywords(use_categories=False)
        
        logger.info(f"Analyzing {len(self.tracked_keywords)} keywords across {len(df)} documents")
        
        results = []
        
        # Group by year
        years = sorted(df['year'].unique())
        logger.info(f"Years to analyze: {years}")
        
        for year in years:
            year_data = df[df['year'] == year]
            
            # Combine all tokens for the year
            year_tokens = []
            for tokens in year_data['tokens']:
                if isinstance(tokens, list):
                    year_tokens.extend(tokens)
            
            total_words = len(year_tokens)
            
            if total_words == 0:
                logger.warning(f"No tokens found for year {year}")
                continue
            
            # Convert to Counter for efficient counting
            token_counter = Counter(year_tokens)
            
            # Count each keyword/category
            for keyword in self.tracked_keywords:
                count = 0
                
                if keyword in self.default_keywords:
                    # It's a category - count all related words
                    related_words = self.default_keywords[keyword]
                    for word in related_words:
                        count += token_counter.get(word.lower(), 0)
                else:
                    # Single keyword
                    count = token_counter.get(keyword.lower(), 0)
                
                # Calculate relative frequency (per 10,000 words)
                frequency = (count / total_words * 10000) if total_words > 0 else 0
                
                results.append({
                    'year': int(year),
                    'keyword': keyword,
                    'raw_count': count,
                    'frequency': round(frequency, 2),
                    'total_words': total_words,
                    'documents': len(year_data)
                })
        
        if not results:
            logger.error("No results generated")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Log the shape and columns
        logger.info(f"Results shape: {results_df.shape}")
        logger.info(f"Results columns: {results_df.columns.tolist()}")
        
        # Add year-over-year change metrics
        if not results_df.empty and 'keyword' in results_df.columns:
            results_df = self._add_change_metrics(results_df)
            
            # Add trend classification
            results_df = self._classify_trends(results_df)
        
        self.analysis_results = results_df
        return results_df
    
    def _add_change_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add year-over-year change metrics"""
        if df.empty or 'keyword' not in df.columns:
            logger.warning("Cannot add change metrics - DataFrame is empty or missing 'keyword' column")
            return df
        
        try:
            df = df.sort_values(['keyword', 'year'])
            
            # Calculate YoY change
            df['yoy_change'] = df.groupby('keyword')['frequency'].pct_change() * 100
            df['yoy_change'] = df['yoy_change'].round(1)
            
            # Calculate moving average
            df['ma_3year'] = df.groupby('keyword')['frequency'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            ).round(2)
        except Exception as e:
            logger.error(f"Error adding change metrics: {e}")
        
        return df
    
    def _classify_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify trend patterns for each keyword"""
        if df.empty or 'keyword' not in df.columns:
            logger.warning("Cannot classify trends - DataFrame is empty or missing 'keyword' column")
            return df
        
        try:
            for keyword in df['keyword'].unique():
                keyword_data = df[df['keyword'] == keyword].sort_values('year')
                
                if len(keyword_data) < 3:
                    trend = 'insufficient_data'
                else:
                    # Compare first third with last third
                    n = len(keyword_data)
                    first_third = keyword_data.head(n // 3)['frequency'].mean()
                    last_third = keyword_data.tail(n // 3)['frequency'].mean()
                    
                    # Calculate trend strength
                    if last_third > first_third * 1.5:
                        trend = 'strong_growth'
                    elif last_third > first_third * 1.1:
                        trend = 'moderate_growth'
                    elif last_third < first_third * 0.5:
                        trend = 'strong_decline'
                    elif last_third < first_third * 0.9:
                        trend = 'moderate_decline'
                    else:
                        trend = 'stable'
                
                df.loc[df['keyword'] == keyword, 'trend_category'] = trend
        except Exception as e:
            logger.error(f"Error classifying trends: {e}")
            df['trend_category'] = 'unknown'
        
        return df
    
    def get_emerging_terms(self, df: pd.DataFrame, 
                           recent_years: int = 3,
                           min_growth: float = 50.0) -> List[Dict]:
        """
        Identify emerging terms showing significant recent growth
        
        Args:
            df: Analysis results DataFrame
            recent_years: Number of recent years to consider
            min_growth: Minimum growth percentage to be considered emerging
        
        Returns:
            List of emerging terms with metrics
        """
        if self.analysis_results is None or self.analysis_results.empty:
            logger.warning("No analysis results available")
            return []
        
        emerging = []
        
        try:
            current_year = self.analysis_results['year'].max()
            
            for keyword in self.tracked_keywords:
                keyword_data = self.analysis_results[
                    self.analysis_results['keyword'] == keyword
                ].sort_values('year')
                
                if keyword_data.empty:
                    continue
                
                # Get recent vs historical data
                recent = keyword_data[keyword_data['year'] > current_year - recent_years]
                historical = keyword_data[keyword_data['year'] <= current_year - recent_years]
                
                if len(recent) > 0 and len(historical) > 0:
                    recent_avg = recent['frequency'].mean()
                    historical_avg = historical['frequency'].mean()
                    
                    if historical_avg > 0:
                        growth = ((recent_avg - historical_avg) / historical_avg) * 100
                        
                        if growth >= min_growth:
                            emerging.append({
                                'keyword': keyword,
                                'growth_rate': round(growth, 1),
                                'recent_frequency': round(recent_avg, 2),
                                'historical_frequency': round(historical_avg, 2),
                                'trend': keyword_data['trend_category'].iloc[-1] if 'trend_category' in keyword_data.columns else 'unknown'
                            })
        except Exception as e:
            logger.error(f"Error identifying emerging terms: {e}")
        
        return sorted(emerging, key=lambda x: x['growth_rate'], reverse=True)
    
    def get_declining_terms(self, df: pd.DataFrame,
                           recent_years: int = 3,
                           min_decline: float = -30.0) -> List[Dict]:
        """
        Identify terms showing significant decline
        """
        if self.analysis_results is None or self.analysis_results.empty:
            logger.warning("No analysis results available")
            return []
        
        declining = []
        
        try:
            current_year = self.analysis_results['year'].max()
            
            for keyword in self.tracked_keywords:
                keyword_data = self.analysis_results[
                    self.analysis_results['keyword'] == keyword
                ].sort_values('year')
                
                if keyword_data.empty:
                    continue
                
                recent = keyword_data[keyword_data['year'] > current_year - recent_years]
                historical = keyword_data[keyword_data['year'] <= current_year - recent_years]
                
                if len(recent) > 0 and len(historical) > 0:
                    recent_avg = recent['frequency'].mean()
                    historical_avg = historical['frequency'].mean()
                    
                    if historical_avg > 0:
                        change = ((recent_avg - historical_avg) / historical_avg) * 100
                        
                        if change <= min_decline:
                            declining.append({
                                'keyword': keyword,
                                'decline_rate': round(change, 1),
                                'recent_frequency': round(recent_avg, 2),
                                'historical_frequency': round(historical_avg, 2),
                                'trend': keyword_data['trend_category'].iloc[-1] if 'trend_category' in keyword_data.columns else 'unknown'
                            })
        except Exception as e:
            logger.error(f"Error identifying declining terms: {e}")
        
        return sorted(declining, key=lambda x: x['decline_rate'])
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """
        Generate automated insights from the analysis
        
        Returns:
            Dictionary of insights
        """
        insights = {
            'summary': {},
            'emerging_terms': [],
            'declining_terms': [],
            'stable_terms': [],
            'volatile_terms': [],
            'recommendations': []
        }
        
        try:
            if self.analysis_results is None or self.analysis_results.empty:
                logger.warning("No analysis results to generate insights from")
                insights['summary']['error'] = "No analysis results available"
                return insights
            
            # Get emerging and declining terms
            insights['emerging_terms'] = self.get_emerging_terms(df)
            insights['declining_terms'] = self.get_declining_terms(df)
            
            # Identify stable terms
            if 'trend_category' in self.analysis_results.columns:
                stable = self.analysis_results[
                    self.analysis_results['trend_category'] == 'stable'
                ]['keyword'].unique().tolist()
                insights['stable_terms'] = stable
            
            # Identify volatile terms (high standard deviation)
            for keyword in self.tracked_keywords:
                keyword_data = self.analysis_results[
                    self.analysis_results['keyword'] == keyword
                ]
                if not keyword_data.empty and len(keyword_data) > 1:
                    mean_freq = keyword_data['frequency'].mean()
                    std_freq = keyword_data['frequency'].std()
                    if mean_freq > 0 and std_freq > mean_freq * 0.5:
                        insights['volatile_terms'].append(keyword)
            
            # Generate recommendations
            if insights['emerging_terms']:
                top_emerging = insights['emerging_terms'][0]
                insights['recommendations'].append(
                    f"Focus on '{top_emerging['keyword']}' - showing {top_emerging['growth_rate']}% growth"
                )
            
            if insights['declining_terms']:
                top_declining = insights['declining_terms'][0]
                insights['recommendations'].append(
                    f"Re-evaluate messaging around '{top_declining['keyword']}' - declining {abs(top_declining['decline_rate'])}%"
                )
            
            # Summary statistics
            insights['summary'] = {
                'total_keywords_tracked': len(self.tracked_keywords),
                'years_analyzed': f"{df['year'].min()}-{df['year'].max()}" if not df.empty else "N/A",
                'total_documents': len(df) if not df.empty else 0,
                'avg_words_per_year': round(
                    self.analysis_results.groupby('year')['total_words'].first().mean()
                ) if not self.analysis_results.empty else 0
            }
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights['summary']['error'] = str(e)
        
        return insights
    
    def export_results(self, output_path: str = "keyword_analysis_results.csv"):
        """Export analysis results to CSV"""
        if self.analysis_results is not None and not self.analysis_results.empty:
            self.analysis_results.to_csv(output_path, index=False)
            logger.info(f"Results exported to {output_path}")
        else:
            logger.warning("No results to export. Run analysis first.")


# Convenience functions
def run_keyword_analysis(df: pd.DataFrame, 
                        keywords: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete keyword analysis pipeline
    
    Args:
        df: Processed DataFrame with tokens
        keywords: Optional custom keywords
    
    Returns:
        Tuple of (results DataFrame, insights dictionary)
    """
    analyzer = KeywordAnalyzer()
    
    # Use default marketing keywords if none provided
    if keywords is None:
        keywords = ['innovation', 'ai', 'privacy', 'sustainability', 'cloud', 
                   'digital', 'customer', 'platform', 'data', 'transformation']
    
    # Run analysis
    results = analyzer.analyze_keyword_trends(df, keywords)
    insights = analyzer.generate_insights(df)
    
    # Log key findings
    if not results.empty:
        logger.info("\n📊 Key Findings:")
        logger.info(f"Emerging terms: {[t['keyword'] for t in insights['emerging_terms'][:3]]}")
        logger.info(f"Declining terms: {[t['keyword'] for t in insights['declining_terms'][:3]]}")
        logger.info(f"Stable terms: {insights['stable_terms'][:3]}")
    
    return results, insights


if __name__ == "__main__":
    # Test the keyword analyzer
    # Create sample data
    sample_data = pd.DataFrame({
        'year': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023],
        'tokens': [
            ['innovation', 'cloud', 'data', 'customer', 'privacy'] * 10,
            ['ai', 'machine', 'learning', 'innovation'] * 15,
            ['sustainability', 'cloud', 'platform', 'digital'] * 12,
            ['ai', 'intelligence', 'automation', 'data'] * 20,
            ['metaverse', 'virtual', 'ai', 'innovation'] * 8,
            ['sustainability', 'carbon', 'renewable', 'environment'] * 25,
            ['ai', 'machine', 'learning', 'automation'] * 30,
            ['privacy', 'security', 'data', 'protection'] * 18
        ]
    })
    
    # Run analysis
    analyzer = KeywordAnalyzer()
    results = analyzer.analyze_keyword_trends(sample_data)
    insights = analyzer.generate_insights(sample_data)
    
    print("Analysis Results:")
    print(results.head(10))
    print("\nInsights:")
    print(f"Emerging: {insights['emerging_terms']}")
    print(f"Declining: {insights['declining_terms']}")