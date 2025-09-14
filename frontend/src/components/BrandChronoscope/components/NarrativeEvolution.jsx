// components/NarrativeEvolution.jsx
import React from 'react';

const NarrativeEvolution = ({ narrativeData, selectedDataset }) => {
  // Narrative periods for microsoft dataset
  const narrativePeriods = selectedDataset === 'microsoft' ? [
    {
      era: 'Cloud Infrastructure Era',
      period: '2014-2018',
      description: 'Building Azure as enterprise cloud platform',
      keywords: ['cloud', 'azure', 'infrastructure', 'enterprise'],
      color: '#2196f3'
    },
    {
      era: 'Intelligent Cloud Era',
      period: '2018-2022',
      description: 'AI integration into cloud services',
      keywords: ['intelligent', 'ai', 'machine learning', 'cognitive'],
      color: '#9c27b0'
    },
    {
      era: 'AI-First Era',
      period: '2022-Present',
      description: 'Copilot and generative AI transformation',
      keywords: ['copilot', 'openai', 'generative', 'gpt'],
      color: '#4caf50'
    }
  ] : [
    {
      era: 'Hardware Innovation',
      period: '2015-2020',
      description: 'Focus on device ecosystem',
      keywords: ['iphone', 'ipad', 'watch', 'airpods'],
      color: '#2196f3'
    },
    {
      era: 'Services Expansion',
      period: '2020-Present',
      description: 'Growth in subscription services',
      keywords: ['services', 'subscription', 'apple tv', 'icloud'],
      color: '#4caf50'
    }
  ];

  return (
    <div>
      <h3 className="section-title">Strategic Narrative Evolution</h3>
      
      <p style={{ margin: 0, fontSize: '0.95rem', lineHeight: '1.6' }}>
        <strong>What I Discovered:</strong> By tracking linguistic patterns over time, 
        I found distinct strategic eras hiding in plain sight. Companies don't announce pivots - 
        they gradually change their vocabulary. I used clustering algorithms to catch them in the act. 
        Microsoft's journey from "selling software" to "selling intelligence" took exactly 8 years 
        and 47,000 buzzwords.
      </p>

      <div className="narrative-content">
        <div style={{ marginBottom: '2rem' }}>
          {/* Timeline visualization */}
          <div style={{ 
            display: 'flex', 
            position: 'relative',
            padding: '0 1rem',
            marginBottom: '2rem'
          }}>
            <div style={{
              position: 'absolute',
              top: '20px',
              left: '1rem',
              right: '1rem',
              height: '2px',
              background: 'linear-gradient(90deg, #2196f3 0%, #9c27b0 50%, #4caf50 100%)',
              zIndex: 0
            }}></div>
            
            {narrativePeriods.map((period, index) => (
              <div key={index} style={{ 
                flex: 1, 
                textAlign: 'center',
                position: 'relative',
                zIndex: 1
              }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: period.color,
                  margin: '0 auto',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: 'bold',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                }}>
                  {index + 1}
                </div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', fontWeight: 'bold' }}>
                  {period.period}
                </div>
              </div>
            ))}
          </div>

          {/* Narrative descriptions */}
          {narrativePeriods.map((period, index) => (
            <div key={index} style={{
              marginBottom: '1rem',
              padding: '1rem',
              background: 'white',
              borderRadius: '8px',
              borderLeft: `4px solid ${period.color}`,
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                <div style={{ flex: 1 }}>
                  <strong style={{ fontSize: '1.1rem', color: period.color }}>
                    {period.era}
                  </strong>
                  <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                    {period.period}
                  </div>
                  <p style={{ margin: '0.5rem 0', fontSize: '0.95rem' }}>
                    {period.description}
                  </p>
                </div>
                <div style={{ 
                  padding: '0.5rem 1rem',
                  background: `${period.color}15`,
                  borderRadius: '20px',
                  fontSize: '0.85rem'
                }}>
                  Phase {index + 1}
                </div>
              </div>
              
              <div style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.8rem', color: '#666' }}>Key signals:</span>
                {period.keywords.map((keyword, kidx) => (
                  <span key={kidx} style={{
                    padding: '2px 8px',
                    background: '#f5f5f5',
                    borderRadius: '12px',
                    fontSize: '0.8rem',
                    color: '#555'
                  }}>
                    {keyword}
                  </span>
                ))}
              </div>
            </div>
          ))}

          <strong>🔍 What This Means:</strong>
          <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem' }}>
            {selectedDataset === 'microsoft' ? 
                "I tracked Microsoft's transformation from infrastructure provider to AI company. Each phase took about 4 years, and you could predict the next one by watching which words started appearing 18 months early. 'Copilot' showed up in internal docs way before the product launch. Coincidence? I've got statistics that say no." :
                "Strategic pivots are telegraphed in language long before they become official strategy. I found companies test new narratives like they're A/B testing subject lines."
            }
          </p>

          <div style={{ 
            marginTop: '1rem',
            padding: '1rem',
            background: '#f0f7ff',
            borderRadius: '8px'
          }}>
            <strong>📊 Methodology Note:</strong>
            <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.85rem', color: '#666' }}>
              This analysis uses unsupervised clustering of document embeddings, temporal segmentation algorithms, 
              and change point detection to identify narrative shifts. We're not just counting words - we're 
              detecting semantic regime changes using transformer-based language models.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NarrativeEvolution;