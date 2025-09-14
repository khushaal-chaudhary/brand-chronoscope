// BrandChronoscope.jsx
import React, { useState, useEffect, useCallback } from 'react';
import './BrandChronoscope.css';

// Import child components (we'll create these next)
import Header from './components/Header';
import DataSourceSelector from './components/DataSourceSelector';
import TabNavigation from './components/TabNavigation';
import KeywordAnalysis from './components/KeywordAnalysis';
import TopicDiscovery from './components/TopicDiscovery';
import NarrativeEvolution from './components/NarrativeEvolution';
import StrategicInsights from './components/StrategicInsights';
import SemanticDrift from './components/SemanticDrift';

const BrandChronoscope = () => {
  // State management
  const [activeTab, setActiveTab] = useState('keyword');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [keywordData, setKeywordData] = useState([]);
  const [topicData, setTopicData] = useState([]);
  const [emergingTerms, setEmergingTerms] = useState([]);
  const [decliningTerms, setDecliningTerms] = useState([]);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [narrativeData, setNarrativeData] = useState(null);
  const [semanticDriftData, setSemanticDriftData] = useState(null);

  const API_URL = 'http://127.0.0.1:8000';

  // Fetch available datasets on mount
  useEffect(() => {
    fetchAvailableDatasets();
  }, []);

  const fetchAvailableDatasets = async () => {
    try {
      const response = await fetch(`${API_URL}/api/datasets`);
      const data = await response.json();
      setAvailableDatasets(data.datasets || []);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const loadDataset = async (datasetName) => {
    setIsLoading(true);
    try {
      const loadResponse = await fetch(`${API_URL}/api/load-dataset/${datasetName}`);
      const loadData = await loadResponse.json();
      setDatasetInfo(loadData);

      // Load keyword analysis
      const analysisResponse = await fetch(`${API_URL}/api/analyze-dataset?dataset_name=${datasetName}&analysis_type=keyword`, {
        method: 'POST'
      });
      const analysisData = await analysisResponse.json();
      
      if (analysisData.status === 'success') {
        processAnalysisData(analysisData.data);
      }

      // Load topic analysis
      const topicResponse = await fetch(`${API_URL}/api/analyze-dataset?dataset_name=${datasetName}&analysis_type=topic`, {
        method: 'POST'
      });
      const topicResult = await topicResponse.json();
      if (topicResult.status === 'success') {
        setTopicData(topicResult.data.topics || []);
      }

      // Load narrative evolution
      const narrativeResponse = await fetch(`${API_URL}/api/analyze-dataset?dataset_name=${datasetName}&analysis_type=narrative`, {
        method: 'POST'
      });
      const narrativeResult = await narrativeResponse.json();
      if (narrativeResult.status === 'success') {
        setNarrativeData(narrativeResult.data);
      }

      // Load semantic drift
      const semanticResponse = await fetch(`${API_URL}/api/analyze-dataset?dataset_name=${datasetName}&analysis_type=semantic-drift`, {
        method: 'POST'
      });
      const semanticResult = await semanticResponse.json();
      if (semanticResult.status === 'success') {
        setSemanticDriftData(semanticResult.data);
      }

    } catch (error) {
      console.error('Error loading dataset:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const processAnalysisData = (data) => {
    if (data.keyword_analysis) {
      const yearMap = {};
      data.keyword_analysis.forEach(item => {
        if (!yearMap[item.year]) {
          yearMap[item.year] = { year: item.year };
        }
        yearMap[item.year][item.keyword] = item.frequency;
      });
      setKeywordData(Object.values(yearMap).sort((a, b) => a.year - b.year));
    }

    if (data.emerging_terms) {
      setEmergingTerms(data.emerging_terms);
    }

    if (data.declining_terms) {
      setDecliningTerms(data.declining_terms);
    }
  };

  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setUploadedFile(file);
    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch(`${API_URL}/api/keyword-analysis`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      if (result.status === 'success') {
        processAnalysisData({ keyword_analysis: result.data });
      }
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleDatasetSelect = (datasetName) => {
    const datasetKey = datasetName.includes('Microsoft') ? 'microsoft' : 
                      datasetName.includes('10-K') ? 'apple-10k' : 'apple-newsroom';
    setSelectedDataset(datasetKey);
    loadDataset(datasetKey);
  };

  // Tab configuration
  const tabs = [
    { 
        id: 'keyword', 
        label: '📊 Keyword Analysis',
        description: 'TF-IDF weighted frequency analysis with statistical trend detection'
    },
    { 
        id: 'topic', 
        label: '🤖 Topic Discovery',
        description: 'Unsupervised clustering using transformer embeddings'
    },
    { 
        id: 'narrative', 
        label: '📈 Narrative Evolution',
        description: 'Temporal segmentation and regime change detection'
    },
    { 
        id: 'insights', 
        label: '🎯 Strategic Insights',
        description: 'Automated insight generation from multi-dimensional analysis'
    },
    { 
        id: 'drift', 
        label: '🔄 Semantic Drift',
        description: 'Word embedding evolution using sentence-transformers'
    }
    ];

  return (
    <div className="chronoscope-container">
      <Header />
      
      <div className="main-content">
        <DataSourceSelector
          availableDatasets={availableDatasets}
          selectedDataset={selectedDataset}
          uploadedFile={uploadedFile}
          datasetInfo={datasetInfo}
          isLoading={isLoading}
          onDatasetSelect={handleDatasetSelect}
          onFileUpload={handleFileUpload}
        />

        {/* Educational Context Panel */}
        {keywordData.length === 0 && !isLoading && (
            <div className="card" style={{ background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' }}>
                <h2 style={{ color: '#2c3e50', marginBottom: '1rem' }}>
                🎯 Corporate Language Evolution Tracker
                </h2>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div>
                    <h3 style={{ color: '#34495e', fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                    What I Built Here
                    </h3>
                    <p style={{ fontSize: '0.9rem', lineHeight: '1.6', color: '#555', marginBottom: '1rem' }}>
                    I tried to answer a simple question: how does corporate language actually evolve? So I turned 10 years 
                    of corporate communications into numbers, then into insights, then into these colorful charts. 
                    Turns out "AI" replaced "cloud" which replaced "digital" which replaced "cyber." 
                    Next year it'll probably be "quantum." I'm just documenting the journey with statistical 
                    significance (p {'<'} 0.05, because standards matter).
                    </p>
                    <ul style={{ fontSize: '0.9rem', lineHeight: '1.6', color: '#555', paddingLeft: '1.2rem' }}>
                        <li>Transformer-based semantic analysis</li>
                        <li>Temporal trend detection with actual statistics</li>
                        <li>Topic discovery without the hallucinations</li>
                        <li>Production ML pipeline that actually works</li>
                    </ul>
                </div>
                
                <div>
                    <h3 style={{ color: '#34495e', fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                    Why This Approach Matters
                    </h3>
                    <p style={{ fontSize: '0.9rem', lineHeight: '1.6', color: '#555', margin: 0 }}>
                    While everyone's racing to prompt-engineer the next chatbot, I figured there's value in 
                    doing what LLMs can't: tracking subtle semantic shifts over time with statistical rigor. 
                    This isn't about generating text - it's about understanding how language evolves. 
                    Think of it as digital archaeology for corporate buzzwords. Turns out Fortune 500 companies 
                    actually care when their competitors start saying "transformation" more than "optimization."
                    </p>
                </div>
                </div>
                
                <div style={{ 
                marginTop: '1.5rem', 
                padding: '1rem', 
                background: 'rgba(255,255,255,0.5)', 
                borderRadius: '8px' 
                }}>
                <strong>🚀 Try It Out:</strong> Load the Microsoft dataset above to watch a decade of strategic pivots unfold. 
                Spoiler: "cloud" peaks around 2018, then "AI" takes over. Nobody saw that coming, right?
                </div>
            </div>
            )}

        {keywordData.length > 0 && (
          <div className="card">
            <TabNavigation
              tabs={tabs}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />

            <div className="tab-content">
              {activeTab === 'keyword' && (
                <KeywordAnalysis
                  keywordData={keywordData}
                  emergingTerms={emergingTerms}
                  decliningTerms={decliningTerms}
                />
              )}

              {activeTab === 'topic' && (
                <TopicDiscovery topicData={topicData} />
              )}

              {activeTab === 'narrative' && (
                <NarrativeEvolution
                  narrativeData={narrativeData}
                  selectedDataset={selectedDataset}
                />
              )}

              {activeTab === 'insights' && (
                <StrategicInsights
                  datasetInfo={datasetInfo}
                  emergingTerms={emergingTerms}
                  decliningTerms={decliningTerms}
                />
              )}

              {activeTab === 'drift' && (
                <SemanticDrift semanticDriftData={semanticDriftData} />
              )}
            </div>
          </div>
        )}

        {keywordData.length === 0 && !isLoading && datasetInfo && (
        <div className="card">
            <div style={{ textAlign: 'center', padding: '3rem' }}>
            <Database size={48} style={{ color: '#667eea', marginBottom: '1rem' }} />
            <h3 style={{ color: '#1e293b', marginBottom: '1rem' }}>
                Hm, That Didn't Work
            </h3>
            <p style={{ color: '#64748b' }}>
                The app tried to analyze the data but something went sideways. 
                Either the format's wrong or you have found a bug which you can tell me about. 
                Try the Microsoft dataset - that one always works.
            </p>
            </div>
        </div>
        )}

        <div className="footer">
            <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                gap: '2rem',
                flexWrap: 'wrap' 
            }}>
                <p style={{ margin: 0 }}>
                Built by Khushaal Chaudary | Data Science & NLP
                </p>
                <div style={{ display: 'flex', gap: '1.5rem' }}>
                <a 
                    href="https://linkedin.com/in/khushaal-chaudhary" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{ 
                    color: 'white', 
                    textDecoration: 'none',
                    opacity: 0.9,
                    transition: 'opacity 0.3s'
                    }}
                    onMouseEnter={(e) => e.target.style.opacity = 1}
                    onMouseLeave={(e) => e.target.style.opacity = 0.9}
                >
                    LinkedIn
                </a>
                <a 
                    href="https://khushaalchaudhary.com" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{ 
                    color: 'white', 
                    textDecoration: 'none',
                    opacity: 0.9,
                    transition: 'opacity 0.3s'
                    }}
                    onMouseEnter={(e) => e.target.style.opacity = 1}
                    onMouseLeave={(e) => e.target.style.opacity = 0.9}
                >
                    About Me
                </a>
                <a 
                    href="https://github.com/khushaal-chaudhary/brand-chronoscope" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{ 
                    color: 'white', 
                    textDecoration: 'none',
                    opacity: 0.9,
                    transition: 'opacity 0.3s'
                    }}
                    onMouseEnter={(e) => e.target.style.opacity = 1}
                    onMouseLeave={(e) => e.target.style.opacity = 0.9}
                >
                    GitHub
                </a>
                </div>
            </div>
            <p style={{ 
                margin: '0.5rem 0 0 0', 
                fontSize: '0.85rem', 
                opacity: 0.7 
            }}>
                React + FastAPI + Transformers | Analyzing corporate language evolution
            </p>
            </div>
      </div>
    </div>
  );
};

export default BrandChronoscope;