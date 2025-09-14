"""
FastAPI Backend with Dataset Support
Serves your existing processed datasets
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import io
import json
from datetime import datetime
from pathlib import Path
import re
from collections import Counter
import sys
from src.semantic_drift_analyzer import SemanticDriftAnalyzer
from src.keyword_clustering import KeywordClusterer

sys.path.append(str(Path(__file__).parent))

import logging
logger = logging.getLogger(__name__)

semantic_analyzer = None

app = FastAPI(
    title="Brand Chronoscope API",
    description="NLP Analysis API with Dataset Support",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded datasets
cached_datasets = {}

@app.get("/")
async def root():
    return {"message": "Brand Chronoscope API is running!"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/datasets")
async def get_available_datasets():
    """Get list of available datasets"""
    datasets = []
    
    # Check for your processed data files
    data_paths = [
        ("Microsoft Shareholder Letters", "data/processed/microsoft_shareholder_letters.csv"),
        ("Apple 10-K Reports", "data/processed/apple_10k_fixed.csv"),
        ("Apple 10-K Reports", "data/processed/apple_10k_pdfs.csv"),
        ("Apple Newsroom", "data/processed/apple_newsroom.csv"),
    ]
    
    for name, path in data_paths:
        if Path(path).exists():
            datasets.append({
                "name": name,
                "path": path,
                "available": True
            })
    
    return {"datasets": datasets}

@app.get("/api/load-dataset/{dataset_name}")
async def load_dataset(dataset_name: str):
    """Load a specific dataset and return basic info"""
    
    dataset_map = {
        "microsoft": "data/processed/microsoft_shareholder_letters.csv",
        "apple-10k": "data/processed/apple_10k_fixed.csv",
        "apple-newsroom": "data/processed/apple_newsroom.csv"
    }
    
    if dataset_name not in dataset_map:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    path = dataset_map[dataset_name]
    if not Path(path).exists():
        # Try alternative path
        if dataset_name == "apple-10k":
            path = "data/processed/apple_10k_pdfs.csv"
        
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
    
    # Load and cache the dataset
    df = pd.read_csv(path)
    cached_datasets[dataset_name] = df
    
    return {
        "status": "success",
        "dataset": dataset_name,
        "rows": len(df),
        "years": sorted(df['year'].unique().tolist()) if 'year' in df.columns else [],
        "columns": df.columns.tolist()
    }

@app.post("/api/analyze-dataset")
async def analyze_dataset(dataset_name: str, analysis_type: str = "keyword", word: Optional[str] = None):
    """Analyze a loaded dataset"""
    
    if dataset_name not in cached_datasets:
        # Try to load it first
        await load_dataset(dataset_name)
    
    df = cached_datasets.get(dataset_name)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    
    if analysis_type == "keyword":
        # Define keywords based on dataset
        if "microsoft" in dataset_name:
            keywords = ['cloud', 'ai', 'azure', 'copilot', 'platform', 'intelligence', 'productivity']
        else:
            keywords = ['innovation', 'ai', 'privacy', 'silicon', 'services', 'ecosystem', 'iphone']
        
        results = calculate_keyword_frequencies(df, keywords)
        
        # Calculate emerging/declining terms
        emerging = identify_emerging_terms(results)
        declining = identify_declining_terms(results)
        
        return {
            "status": "success",
            "data": {
                "keyword_analysis": results,
                "emerging_terms": emerging,
                "declining_terms": declining
            }
        }
    
    elif analysis_type == "topic":
        try:
            clusterer = KeywordClusterer()
            theme_results = clusterer.analyze_theme_evolution(df)
            
            # Format results for frontend
            topics_by_year = []
            if theme_results and 'keywords_by_year' in theme_results:
                for year, keywords in theme_results['keywords_by_year'].items():
                    # Get top meaningful keywords (filter out short ones)
                    meaningful_keywords = []
                    for keyword, score in keywords[:10]:  # Top 10 keywords
                        if len(keyword) > 3 and not keyword.isdigit():
                            meaningful_keywords.append({
                                "word": keyword,
                                "count": round(score * 100)  # Convert score to pseudo-count
                            })
                    
                    if meaningful_keywords:
                        topics_by_year.append({
                            "year": year,
                            "topics": meaningful_keywords[:5]  # Top 5 per year
                        })
            
            return {
                "status": "success",
                "data": {
                    "topics": topics_by_year,
                    "themes": theme_results.get('themes', {}),
                    "emerging": theme_results.get('emerging', [])
                }
            }
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            # Fallback to simple topic modeling
            topics = simple_topic_modeling(df)
            return {
                "status": "success",
                "data": {"topics": topics}
            }
    
    elif analysis_type == "narrative":
        # Narrative evolution (if available)
        narrative_path = Path("data/processed/microsoft_narrative_evolution.csv")
        if narrative_path.exists() and "microsoft" in dataset_name:
            narrative_df = pd.read_csv(narrative_path)
            return {
                "status": "success",
                "data": {
                    "narrative_evolution": narrative_df.to_dict('records')
                }
            }
        return {"status": "error", "message": "Narrative data not available"}
    
    elif analysis_type == "semantic-drift":
        word_to_analyze = word or "platform"
        
        # Initialize analyzer if not already done (singleton pattern for performance)
        global semantic_analyzer
        if semantic_analyzer is None:
            logger.info("Initializing semantic drift analyzer...")
            semantic_analyzer = SemanticDriftAnalyzer()
        
        try:
            # Run the actual analysis
            logger.info(f"Analyzing semantic drift for '{word_to_analyze}'")
            result = semantic_analyzer.analyze_word_evolution(df, word_to_analyze)
            
            # Format the response
            data = {
                "word": result['word'],
                "drift_score": round(result['total_drift_score'], 3),
                "past_context": "Not available",  # Default
                "present_context": "Not available"  # Default
            }
            
            # Extract contexts if available
            if result['contexts_by_year']:
                years = sorted(result['contexts_by_year'].keys())
                if len(years) >= 2:
                    # Get first and last year contexts
                    data["past_context"] = result['contexts_by_year'][years[0]][:100] + "..."
                    data["present_context"] = result['contexts_by_year'][years[-1]][:100] + "..."
            
            # Add semantic neighbors if available
            if result['semantic_neighbors_by_year']:
                years = sorted(result['semantic_neighbors_by_year'].keys())
                if years:
                    early_neighbors = result['semantic_neighbors_by_year'].get(years[0], [])
                    recent_neighbors = result['semantic_neighbors_by_year'].get(years[-1], [])
                    
                    if early_neighbors:
                        data["past_context"] = ", ".join(early_neighbors[:5])
                    if recent_neighbors:
                        data["present_context"] = ", ".join(recent_neighbors[:5])
            
            return {
                "status": "success",
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Error in semantic drift analysis: {e}")
            # Fallback to mock data if analysis fails
            return {
                "status": "error",
                "message": str(e),
                "data": {
                    "word": word_to_analyze,
                    "drift_score": 0.0,
                    "past_context": "Analysis failed",
                    "present_context": "Analysis failed"
                }
            }
    
    else:
        raise HTTPException(status_code=400, detail="Invalid analysis type")

@app.post("/api/keyword-analysis")
async def keyword_analysis(file: Optional[UploadFile] = None, dataset: Optional[str] = None):
    """Analyze keywords from uploaded file or dataset"""
    
    if file:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    elif dataset and dataset in cached_datasets:
        df = cached_datasets[dataset]
    else:
        raise HTTPException(status_code=400, detail="No data source provided")
    
    # Default keywords
    keywords = ["innovation", "cloud", "ai", "platform", "security", "digital", "transformation"]
    
    results = calculate_keyword_frequencies(df, keywords)
    
    return {"status": "success", "data": results}

def calculate_keyword_frequencies(df: pd.DataFrame, keywords: List[str]) -> List[Dict]:
    """Calculate keyword frequencies by year"""
    
    results = []
    
    for year in sorted(df['year'].unique()):
        year_text = ' '.join(df[df['year'] == year]['text'].astype(str).tolist()).lower()
        total_words = len(year_text.split())
        
        for keyword in keywords:
            count = len(re.findall(r'\b' + keyword.lower() + r'\b', year_text))
            frequency = (count / total_words * 10000) if total_words > 0 else 0
            
            results.append({
                "year": int(year),
                "keyword": keyword,
                "frequency": round(frequency, 2),
                "count": count
            })
    
    return results

def identify_emerging_terms(keyword_data: List[Dict]) -> List[Dict]:
    """Identify emerging terms from keyword data"""
    
    # Group by keyword
    keyword_trends = {}
    for item in keyword_data:
        keyword = item['keyword']
        if keyword not in keyword_trends:
            keyword_trends[keyword] = []
        keyword_trends[keyword].append((item['year'], item['frequency']))
    
    emerging = []
    for keyword, yearly_data in keyword_trends.items():
        yearly_data.sort(key=lambda x: x[0])
        
        if len(yearly_data) >= 3:
            # Compare first half with second half
            mid = len(yearly_data) // 2
            early_data = yearly_data[:mid]
            recent_data = yearly_data[mid:]
            
            # Calculate averages, handling zero values
            early_avg = np.mean([freq for _, freq in early_data]) if early_data else 0
            recent_avg = np.mean([freq for _, freq in recent_data]) if recent_data else 0
            
            # Special handling for new terms (copilot, etc.)
            if keyword.lower() in ['copilot', 'openai', 'gpt', 'generative']:
                # Check if it's truly new (appears only in recent years)
                first_appearance = yearly_data[0][0]
                if first_appearance >= 2020:  # New term
                    # Calculate growth from first appearance
                    if len(yearly_data) > 1:
                        first_freq = yearly_data[0][1]
                        last_freq = yearly_data[-1][1]
                        if first_freq > 0:
                            growth = ((last_freq - first_freq) / first_freq) * 100
                        else:
                            growth = 100 if last_freq > 0 else 0  # New term appearance
                    else:
                        growth = 100  # Brand new term
                    
                    emerging.append({
                        "keyword": keyword,
                        "growth_rate": round(growth),
                        "recent_frequency": round(recent_avg, 1) if recent_avg > 0 else round(yearly_data[-1][1], 1)
                    })
                    continue
            
            # Standard calculation for established terms
            if early_avg > 0 and recent_avg > early_avg * 1.5:  # 50% increase
                growth = ((recent_avg - early_avg) / early_avg) * 100
                emerging.append({
                    "keyword": keyword,
                    "growth_rate": round(growth),
                    "recent_frequency": round(recent_avg, 1)
                })
            elif early_avg == 0 and recent_avg > 0:  # New term
                emerging.append({
                    "keyword": keyword,
                    "growth_rate": 100,  # Show as 100% for new terms
                    "recent_frequency": round(recent_avg, 1)
                })
    
    return sorted(emerging, key=lambda x: x['growth_rate'], reverse=True)[:3]

def identify_declining_terms(keyword_data: List[Dict]) -> List[Dict]:
    """Identify declining terms from keyword data"""
    
    keyword_trends = {}
    for item in keyword_data:
        keyword = item['keyword']
        if keyword not in keyword_trends:
            keyword_trends[keyword] = []
        keyword_trends[keyword].append((item['year'], item['frequency']))
    
    declining = []
    for keyword, yearly_data in keyword_trends.items():
        yearly_data.sort(key=lambda x: x[0])
        
        if len(yearly_data) >= 3:
            mid = len(yearly_data) // 2
            early_avg = np.mean([freq for _, freq in yearly_data[:mid]])
            recent_avg = np.mean([freq for _, freq in yearly_data[mid:]])
            
            if recent_avg < early_avg * 0.7:  # 30% decrease
                decline = ((recent_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
                declining.append({
                    "keyword": keyword,
                    "decline_rate": round(decline),
                    "recent_frequency": round(recent_avg, 1)
                })
    
    return sorted(declining, key=lambda x: x['decline_rate'])[:3]

def simple_topic_modeling(df: pd.DataFrame) -> List[Dict]:
    """Simple topic modeling using word frequency"""
    
    topics_by_year = []
    
    for year in sorted(df['year'].unique())[-5:]:  # Last 5 years
        year_text = ' '.join(df[df['year'] == year]['text'].astype(str).tolist()).lower()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        
        words = [w for w in year_text.split() if w not in stop_words and len(w) > 3]
        word_freq = Counter(words)
        
        # Get top topics
        top_words = word_freq.most_common(5)
        
        topics_by_year.append({
            "year": int(year),
            "topics": [{"word": word, "count": count} for word, count in top_words]
        })
    
    return topics_by_year

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)