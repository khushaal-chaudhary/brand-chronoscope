"""
Enhanced Topic Modeling for Small Document Sets
Splits documents into chunks for better topic discovery
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError as e:
    BERTOPIC_AVAILABLE = False
    logger.error(f"Missing dependencies: {e}")

class EnhancedTopicModeler:
    """
    Enhanced topic modeling for small corporate document sets
    """
    
    def __init__(self):
        """Initialize with optimized parameters for small datasets"""
        self.topic_model = None
        self.documents_with_metadata = []
        
        if BERTOPIC_AVAILABLE:
            logger.info("Initializing enhanced topic model...")
            
            # Use a better embedding model
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Custom UMAP for small datasets
            umap_model = UMAP(
                n_neighbors=3,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            # Custom HDBSCAN for small datasets
            hdbscan_model = HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Enhanced stop words list
            custom_stop_words = list(CountVectorizer(stop_words='english').get_stop_words())
            custom_stop_words.extend([
                # Generic adjectives
                'new', 'old', 'good', 'bad', 'best', 'better', 'worse', 'worst',
                'big', 'small', 'large', 'little', 'great', 'major', 'minor',
                'high', 'low', 'long', 'short', 'many', 'much', 'few', 'several',
                # Time related
                'year', 'years', 'month', 'months', 'day', 'days', 'time', 'times',
                'fiscal', 'annual', 'quarter', 'period', 'date', 'today', 'tomorrow',
                # Generic business terms
                'company', 'business', 'corporation', 'million', 'billion', 'thousand',
                'percent', 'number', 'amount', 'total', 'level', 'rate',
                # Common verbs that don't add meaning
                'make', 'made', 'making', 'help', 'helped', 'helping', 'become',
                'provide', 'provided', 'providing', 'include', 'including', 'included',
                'continue', 'continued', 'continuing', 'remain', 'remained',
                # Document words
                'page', 'section', 'table', 'figure', 'note', 'item', 'part'
            ])
            
            # Custom vectorizer with enhanced stop words
            vectorizer_model = CountVectorizer(
                stop_words=custom_stop_words,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 3),
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
            )
            
            # Initialize BERTopic with custom components
            self.topic_model = BERTopic(
                embedding_model=self.sentence_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                min_topic_size=2,
                nr_topics="auto",
                verbose=True
            )
    
    def chunk_documents(self, df: pd.DataFrame, chunk_size: int = 1000) -> List[Tuple[str, Dict]]:
        """
        Split documents into chunks for better topic modeling
        
        Args:
            df: DataFrame with 'year' and 'text' columns
            chunk_size: Approximate size of each chunk in words
            
        Returns:
            List of (text_chunk, metadata) tuples
        """
        chunks = []
        
        for _, row in df.iterrows():
            year = row['year']
            text = row['text']
            
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            # Group sentences into chunks
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                words = sentence.split()
                current_chunk.append(sentence)
                current_size += len(words)
                
                if current_size >= chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) > 100:  # Minimum chunk size
                        chunks.append((
                            chunk_text,
                            {'year': year, 'chunk_id': len(chunks)}
                        ))
                    current_chunk = []
                    current_size = 0
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 100:
                    chunks.append((
                        chunk_text,
                        {'year': year, 'chunk_id': len(chunks)}
                    ))
        
        logger.info(f"Created {len(chunks)} chunks from {len(df)} documents")
        return chunks
    
    def extract_meaningful_sections(self, text: str) -> List[str]:
        """
        Extract meaningful sections from 10-K text
        Focus on strategic language, not boilerplate
        """
        sections = []
        
        # Patterns for meaningful content
        strategic_patterns = [
            r"(?i)(our strategy|strategic priorities|key initiatives).*?(?=[A-Z][a-z]+:|\n\n|$)",
            r"(?i)(competitive advantages?|differentiators?).*?(?=[A-Z][a-z]+:|\n\n|$)",
            r"(?i)(future|outlook|forward[- ]looking).*?(?=[A-Z][a-z]+:|\n\n|$)",
            r"(?i)(innovation|research and development|R&D).*?(?=[A-Z][a-z]+:|\n\n|$)",
            r"(?i)(artificial intelligence|AI|machine learning|ML).*?(?=[A-Z][a-z]+:|\n\n|$)",
            r"(?i)(sustainability|environmental|carbon neutral).*?(?=[A-Z][a-z]+:|\n\n|$)",
            r"(?i)(privacy|security|data protection).*?(?=[A-Z][a-z]+:|\n\n|$)",
            r"(?i)(services|ecosystem|platform).*?(?=[A-Z][a-z]+:|\n\n|$)",
        ]
        
        for pattern in strategic_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if len(match) > 200 and len(match) < 5000:
                    sections.append(match)
        
        return sections
    
    def discover_topics_enhanced(self, df: pd.DataFrame) -> Dict:
        """
        Enhanced topic discovery for small document sets
        """
        if not BERTOPIC_AVAILABLE:
            return {"error": "BERTopic not installed"}
        
        logger.info("Starting enhanced topic discovery...")
        
        # Method 1: Chunk documents for more data points
        chunks = self.chunk_documents(df, chunk_size=500)
        
        if len(chunks) < 10:
            logger.warning(f"Only {len(chunks)} chunks created. Trying smaller chunks...")
            chunks = self.chunk_documents(df, chunk_size=200)
        
        # Extract texts and metadata
        texts = [chunk[0] for chunk in chunks]
        metadata = [chunk[1] for chunk in chunks]
        
        logger.info(f"Processing {len(texts)} text chunks...")
        
        try:
            # Fit the model
            topics, probs = self.topic_model.fit_transform(texts)
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            # Analyze topics by year
            topics_by_year = self._analyze_temporal_topics(topics, metadata)
            
            # Get better topic labels
            topic_labels = self._generate_topic_labels()
            
            # Find emerging themes
            emerging_themes = self._identify_emerging_themes(topics, metadata)
            
            logger.info(f"✅ Discovered {len(topic_info)-1} topics from {len(texts)} chunks")
            
            return {
                "success": True,
                "num_topics": len(topic_info) - 1,
                "num_chunks": len(texts),
                "topic_info": topic_info,
                "topic_labels": topic_labels,
                "topics_by_year": topics_by_year,
                "emerging_themes": emerging_themes
            }
            
        except Exception as e:
            logger.error(f"Error in topic discovery: {e}")
            return {"error": str(e)}
    
    def _analyze_temporal_topics(self, topics: List[int], metadata: List[Dict]) -> Dict:
        """Analyze how topics change over time"""
        topics_by_year = {}
        
        for topic, meta in zip(topics, metadata):
            year = meta['year']
            if year not in topics_by_year:
                topics_by_year[year] = {}
            
            if topic != -1:  # Ignore outliers
                if topic not in topics_by_year[year]:
                    topics_by_year[year][topic] = 0
                topics_by_year[year][topic] += 1
        
        return topics_by_year
    
    def _generate_topic_labels(self) -> Dict:
        """Generate meaningful labels for discovered topics"""
        labels = {}
        
        for topic_id in range(len(self.topic_model.get_topic_info()) - 1):
            if topic_id == -1:
                continue
                
            # Get top words for this topic
            words = self.topic_model.get_topic(topic_id)
            if words:
                # Filter out very common words
                meaningful_words = []
                for word, score in words[:10]:
                    if len(word) > 3 and word not in ['that', 'this', 'have', 'from', 'with', 'will']:
                        meaningful_words.append(word)
                    if len(meaningful_words) >= 3:
                        break
                
                if meaningful_words:
                    labels[topic_id] = ', '.join(meaningful_words[:3]).title()
                else:
                    labels[topic_id] = f"Topic {topic_id}"
        
        return labels
    
    def _identify_emerging_themes(self, topics: List[int], metadata: List[Dict]) -> List[Dict]:
        """Identify topics that are growing over time"""
        # Count topics by year
        year_topic_counts = {}
        
        for topic, meta in zip(topics, metadata):
            if topic == -1:
                continue
                
            year = meta['year']
            if year not in year_topic_counts:
                year_topic_counts[year] = {}
            
            if topic not in year_topic_counts[year]:
                year_topic_counts[year][topic] = 0
            year_topic_counts[year][topic] += 1
        
        # Identify emerging topics
        emerging = []
        years = sorted(year_topic_counts.keys())
        
        if len(years) > 2:
            early_years = years[:len(years)//2]
            recent_years = years[len(years)//2:]
            
            all_topics = set()
            for year_topics in year_topic_counts.values():
                all_topics.update(year_topics.keys())
            
            for topic in all_topics:
                early_count = sum(year_topic_counts.get(y, {}).get(topic, 0) for y in early_years)
                recent_count = sum(year_topic_counts.get(y, {}).get(topic, 0) for y in recent_years)
                
                if recent_count > early_count * 1.5:  # 50% increase
                    emerging.append({
                        'topic': topic,
                        'early_count': early_count,
                        'recent_count': recent_count,
                        'growth': (recent_count - early_count) / max(early_count, 1) * 100
                    })
        
        return sorted(emerging, key=lambda x: x['growth'], reverse=True)

def run_enhanced_analysis():
    """Run enhanced topic analysis on Apple data"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.data.text_processor import TextProcessor
    
    print("\n" + "="*60)
    print("🚀 ENHANCED TOPIC DISCOVERY")
    print("="*60)
    
    # Load data
    data_path = Path("data/processed/apple_10k_fixed.csv")
    if not data_path.exists():
        data_path = Path("data/processed/apple_10k_pdfs.csv")
    
    if not data_path.exists():
        print("❌ No data found")
        return
    
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} documents from {df['year'].min()}-{df['year'].max()}")
    
    # Clean text
    processor = TextProcessor()
    df['text'] = df['text'].apply(lambda x: processor.clean_text(x) if isinstance(x, str) else "")
    
    # Run enhanced analysis
    modeler = EnhancedTopicModeler()
    results = modeler.discover_topics_enhanced(df)
    
    if results.get("success"):
        print(f"\n✨ DISCOVERED {results['num_topics']} TOPICS from {results['num_chunks']} chunks")
        
        print("\n📊 TOPIC LABELS:")
        for topic_id, label in results['topic_labels'].items():
            print(f"  Topic {topic_id}: {label}")
        
        print("\n📈 EMERGING THEMES:")
        for theme in results['emerging_themes'][:3]:
            topic_label = results['topic_labels'].get(theme['topic'], f"Topic {theme['topic']}")
            print(f"  📈 {topic_label}: +{theme['growth']:.0f}% growth")
        
        print("\n📅 TOPICS BY YEAR:")
        for year in sorted(results['topics_by_year'].keys())[-3:]:  # Last 3 years
            print(f"  {year}: {results['topics_by_year'][year]}")
    else:
        print(f"❌ Error: {results.get('error')}")

if __name__ == "__main__":
    run_enhanced_analysis()