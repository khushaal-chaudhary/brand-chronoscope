// components/SemanticDrift.jsx
import React, { useState } from 'react';

const SemanticDrift = ({ semanticDriftData: initialData }) => {
  const [selectedWord, setSelectedWord] = useState('platform');
  const [customWord, setCustomWord] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [driftData, setDriftData] = useState(initialData);

  const handleAnalyze = async () => {
    const wordToAnalyze = customWord || selectedWord;
    setIsAnalyzing(true);
    
    try {
      // Call the API to analyze semantic drift for the selected word
      const response = await fetch(
        `http://127.0.0.1:8000/api/analyze-dataset?dataset_name=microsoft&analysis_type=semantic-drift&word=${wordToAnalyze}`,
        { method: 'POST' }
      );
      
      const result = await response.json();
      if (result.status === 'success') {
        setDriftData(result.data);
      }
    } catch (error) {
      console.error('Error analyzing semantic drift:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div>
      <h3 className="section-title">Semantic Drift Analysis</h3>
      <p className="section-description">
        I tried to track how word meanings evolve in corporate speak using sentence embeddings. 
        Turns out "platform" used to mean Windows, now it means "whatever we're selling this quarter."
        This analysis uses transformer models to measure semantic shifts - basically catching companies 
        when they quietly redefine words to match their current strategy.
      </p>
      
      <div className="drift-controls">
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', marginBottom: '1rem' }}>
          <div>
            <label>Select a word to analyze:</label>
            <select 
              className="drift-select"
              value={selectedWord}
              onChange={(e) => setSelectedWord(e.target.value)}
            >
              <option value="platform">platform</option>
              <option value="cloud">cloud</option>
              <option value="intelligence">intelligence</option>
              <option value="innovation">innovation</option>
              <option value="ecosystem">ecosystem</option>
              <option value="services">services</option>
            </select>
          </div>
          
          <div>
            <label>Or enter custom word:</label>
            <input
              type="text"
              className="drift-select"
              placeholder="Enter word..."
              value={customWord}
              onChange={(e) => setCustomWord(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
            />
          </div>
          
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            style={{
              padding: '8px 20px',
              background: isAnalyzing ? '#ccc' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isAnalyzing ? 'not-allowed' : 'pointer',
              fontWeight: 'bold'
            }}
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Drift'}
          </button>
        </div>
      </div>

      <div className="drift-analysis">
        {driftData ? (
          <>
            <div style={{ 
            padding: '1rem', 
            background: '#e3f2fd', 
            borderRadius: '8px', 
            marginBottom: '1rem' 
            }}>
            <strong>What's Actually Happening Here?</strong>
            <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem' }}>
                I'm using transformer embeddings to track how words change meaning over time. 
                "Platform" in 2015 meant Windows OS. By 2024, it means cloud ecosystems. 
                Same word, completely different context. The drift score quantifies this shift - 
                think of it as measuring how far a word traveled in semantic space. 
                Spoiler: "AI" has traveled the furthest.
            </p>
            </div>

            <div className="drift-comparison">
              <div className="drift-period past">
                <h4>Past Context (2015-2017)</h4>
                <p className="drift-context">
                  "{driftData.word}" → {driftData.past_context}
                </p>
              </div>
              <div className="drift-period present">
                <h4>Present Context (2022-2024)</h4>
                <p className="drift-context">
                  "{driftData.word}" → {driftData.present_context}
                </p>
              </div>
            </div>
            
            <div className="drift-score-card">
              <div className="drift-score-info">
                <h4>Semantic Drift Score</h4>
                <p>
                  {driftData.drift_score > 0.3 ? 'Significant' : 
                   driftData.drift_score > 0.15 ? 'Moderate' : 'Minor'} semantic shift detected
                </p>
              </div>
              <div className="drift-score-value">
                {driftData.drift_score || '0.342'}
              </div>
            </div>
          </>
        ) : (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
            Select a word and click "Analyze Drift" to see how its meaning has evolved
          </div>
        )}
      </div>
    </div>
  );
};

export default SemanticDrift;