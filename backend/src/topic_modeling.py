"""
Dynamic Topic Modeling for Brand Chronoscope
Phase 2 - Step 1: Basic implementation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if BERTopic is available
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
    logger.info("✅ BERTopic loaded successfully")
except ImportError as e:
    BERTOPIC_AVAILABLE = False
    logger.warning(f"❌ BERTopic not available: {e}")

class SimpleTopicModeler:
    """
    Simple implementation of dynamic topic modeling
    Discovers what Apple talks about without predefined keywords
    """
    
    def __init__(self):
        """Initialize the topic modeler"""
        self.topic_model = None
        self.topics = None
        self.topic_info = None
        
        if BERTOPIC_AVAILABLE:
            logger.info("Initializing BERTopic model...")
            # Use a small, fast model for testing
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.topic_model = BERTopic(
                embedding_model=self.sentence_model,
                min_topic_size=2,  # Small for testing with limited docs
                nr_topics="auto",  # Let it discover number of topics
                verbose=True
            )
            logger.info("✅ Topic model initialized")
        else:
            logger.error("Cannot initialize - BERTopic not installed")
    
    def discover_topics(self, df: pd.DataFrame) -> Dict:
        """
        Discover topics in the documents
        
        Args:
            df: DataFrame with 'year' and 'text' columns
            
        Returns:
            Dictionary with topic information
        """
        if not BERTOPIC_AVAILABLE:
            return {"error": "BERTopic not installed"}
        
        logger.info(f"Discovering topics in {len(df)} documents...")
        
        # Prepare documents
        documents = df['text'].tolist()
        
        # Fit the model
        try:
            topics, probs = self.topic_model.fit_transform(documents)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            
            # Get topics per year
            topics_per_year = self._analyze_topics_by_year(df, topics)
            
            # Store results
            self.topics = topics
            self.topic_info = topic_info
            
            logger.info(f"✅ Discovered {len(topic_info)-1} topics")  # -1 for outlier topic
            
            return {
                "success": True,
                "num_topics": len(topic_info) - 1,
                "topic_info": topic_info,
                "topics_per_year": topics_per_year,
                "topic_labels": self._get_topic_labels()
            }
            
        except Exception as e:
            logger.error(f"Error discovering topics: {e}")
            return {"error": str(e)}
    
    def _analyze_topics_by_year(self, df: pd.DataFrame, topics: List[int]) -> Dict:
        """Analyze how topics distribute across years"""
        df['topic'] = topics
        
        # Count topics per year
        topics_by_year = {}
        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]
            topic_counts = year_df['topic'].value_counts().to_dict()
            topics_by_year[int(year)] = topic_counts
        
        return topics_by_year
    
    def _get_topic_labels(self) -> Dict:
        """Get human-readable labels for topics"""
        if self.topic_info is None:
            return {}
        
        labels = {}
        for idx, row in self.topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                # Get top 3 words for the topic
                topic_words = self.topic_model.get_topic(row['Topic'])
                if topic_words:
                    top_words = [word for word, _ in topic_words[:3]]
                    labels[row['Topic']] = f"Topic {row['Topic']}: {', '.join(top_words)}"
        
        return labels
    
    def get_topic_evolution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Track how topics evolve over time
        
        Returns:
            DataFrame showing topic prevalence by year
        """
        if self.topics is None:
            return pd.DataFrame()
        
        # Add topics to dataframe
        df['topic'] = self.topics
        
        # Calculate topic prevalence by year
        evolution = []
        
        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]
            total_docs = len(year_df)
            
            for topic in year_df['topic'].unique():
                if topic != -1:  # Skip outliers
                    count = len(year_df[year_df['topic'] == topic])
                    evolution.append({
                        'year': int(year),
                        'topic': topic,
                        'count': count,
                        'prevalence': count / total_docs * 100
                    })
        
        evolution_df = pd.DataFrame(evolution)
        return evolution_df

# Test function
def test_topic_modeling():
    """Test the topic modeler with sample data"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'year': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
        'text': [
            "Apple Silicon M1 chip revolutionizes Mac performance with incredible efficiency",
            "Privacy labels in App Store increase transparency for users",
            "M1 Pro and Max deliver unprecedented performance for professionals",
            "Privacy is fundamental to everything we build at Apple",
            "Apple Intelligence features expand across our products",
            "Services reach one billion subscriptions milestone",
            "Vision Pro introduces spatial computing platform",
            "Generative AI enhances user experience while maintaining privacy",
            "Apple Intelligence with on-device processing protects user data",
            "Carbon neutral products ship globally meeting environmental goals"
        ]
    })
    
    # Initialize modeler
    modeler = SimpleTopicModeler()
    
    # Discover topics
    results = modeler.discover_topics(sample_data)
    
    if "success" in results and results["success"]:
        print(f"\n✅ Discovered {results['num_topics']} topics")
        print("\n📊 Topic Labels:")
        for topic_id, label in results['topic_labels'].items():
            print(f"  {label}")
        
        print("\n📈 Topics by Year:")
        for year, topics in results['topics_per_year'].items():
            print(f"  {year}: {topics}")
        
        # Get evolution
        evolution = modeler.get_topic_evolution(sample_data)
        if not evolution.empty:
            print("\n📉 Topic Evolution:")
            print(evolution)
    else:
        print(f"❌ Error: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    print("🧪 Testing Topic Modeling...")
    print("-" * 50)
    test_topic_modeling()