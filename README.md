# Brand Chronoscope

Track how corporate language evolves over time using transformer-based NLP and statistical analysis.

## What It Does

Analyzes 10+ years of corporate communications to identify:
- Emerging and declining terminology trends
- Semantic drift in word meanings
- Strategic narrative evolution
- Topic discovery without predefined keywords

## Tech Stack

**Frontend:** React, Recharts, Lucide Icons  
**Backend:** FastAPI, Python 3.12  
**NLP:** Sentence Transformers, Scikit-learn, NLTK  
**Data:** Pandas, NumPy

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
### Frontend
```bash
cd frontend
npm install
npm run dev
```
Navigate to http://localhost:5173

### Features
- Keyword Analysis - TF-IDF weighted frequency tracking with statistical trend detection
- Topic Discovery - Unsupervised clustering using transformer embeddings
- Semantic Drift - Tracks how word meanings change over time (e.g., "platform" in 2015 vs 2024)
- Strategic Insights - Automated detection of narrative shifts and business pivots

### Data Format
Upload CSV files with:
- year column (integer)
- text column (document content)
- Optional: source column

### Example Insights
Using Microsoft shareholder letters (2015-2024):
- "AI" increased 181% while "cloud" declined 58%
- "Platform" semantically drifted from "Windows OS" to "ecosystem/marketplace"
- Strategic pivot from cloud infrastructure to AI-first detected ~18 months before official announcements

### Why This Matters
LLMs generate text. This analyzes how text evolves. Different problem, different approach.
Built for data science portfolios and corporate strategy analysis.

---
Built by [Khushaal Chaudhary](https://khushaalchaudhary.com) | [LinkedIn](https://www.linkedin.com/in/khushaal-chaudhary)