// components/TopicDiscovery.jsx
import React, { useState } from 'react';
import { Brain, TrendingUp, TrendingDown } from 'lucide-react';

const TopicDiscovery = ({ topicData }) => {
  const [viewMode, setViewMode] = useState('heatmap'); // 'heatmap' or 'timeline'
  
  // Prepare data for heatmap
  const prepareHeatmapData = () => {
    if (!topicData || topicData.length === 0) return { words: [], years: [], matrix: [] };
    
    const years = [...new Set(topicData.map(d => d.year))].sort();
    const wordMap = {};
    
    // Collect all words and their frequencies
    topicData.forEach(yearData => {
      yearData.topics?.forEach(topic => {
        if (!wordMap[topic.word]) {
          wordMap[topic.word] = {};
        }
        wordMap[topic.word][yearData.year] = topic.count;
      });
    });
    
    // Get top words by total frequency
    const words = Object.keys(wordMap)
      .sort((a, b) => {
        const totalA = Object.values(wordMap[a]).reduce((sum, val) => sum + val, 0);
        const totalB = Object.values(wordMap[b]).reduce((sum, val) => sum + val, 0);
        return totalB - totalA;
      })
      .slice(0, 12); // Top 12 words
    
    // Create matrix
    const matrix = words.map(word => 
      years.map(year => wordMap[word][year] || 0)
    );
    
    return { words, years, matrix };
  };
  
  const { words, years, matrix } = prepareHeatmapData();
  
  // Calculate color intensity
  const getColor = (value, maxValue) => {
    if (value === 0) return '#f3f4f6';
    const intensity = value / maxValue;
    const opacity = 0.2 + (intensity * 0.8);
    return `rgba(102, 126, 234, ${opacity})`;
  };
  
  const maxValue = Math.max(...matrix.flat());
  
  return (
    <div style={{ width: '100%' }}>
      <h3 className="section-title">AI-Powered Topic Discovery</h3>
      <p className="section-description">
        I skip the keyword guesswork and let algorithms discover the actual narrative. They're simply better at it. 
        This approach caught "metaverse" rising and falling faster than a cryptocurrency pump-and-dump.
      </p>
      
      {topicData && topicData.length > 0 ? (
        <>
          {/* View Mode Selector */}
          <div style={{ 
            display: 'flex', 
            gap: '1rem', 
            marginBottom: '2rem',
            justifyContent: 'center'
          }}>
            <button
              onClick={() => setViewMode('heatmap')}
              style={{
                padding: '8px 20px',
                background: viewMode === 'heatmap' ? '#667eea' : 'white',
                color: viewMode === 'heatmap' ? 'white' : '#667eea',
                border: '2px solid #667eea',
                borderRadius: '20px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Heatmap View
            </button>
            <button
              onClick={() => setViewMode('timeline')}
              style={{
                padding: '8px 20px',
                background: viewMode === 'timeline' ? '#667eea' : 'white',
                color: viewMode === 'timeline' ? 'white' : '#667eea',
                border: '2px solid #667eea',
                borderRadius: '20px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Timeline View
            </button>
          </div>
          
          {viewMode === 'heatmap' ? (
            <div style={{ 
              background: 'white', 
              padding: '2rem', 
              borderRadius: '12px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              overflowX: 'auto'
            }}>
              <h4 style={{ marginBottom: '1.5rem', color: '#2c3e50' }}>
                Topic Intensity Map (Darker = More Frequent)
              </h4>
              
              {/* Heatmap Grid */}
              <div style={{ display: 'inline-block', minWidth: '100%' }}>
                {/* Year headers */}
                <div style={{ display: 'flex', marginBottom: '0.5rem' }}>
                  <div style={{ width: '120px' }}></div>
                  {years.map(year => (
                    <div 
                      key={year}
                      style={{ 
                        width: '80px', 
                        textAlign: 'center',
                        fontWeight: 'bold',
                        fontSize: '0.9rem',
                        color: '#667eea'
                      }}
                    >
                      {year}
                    </div>
                  ))}
                </div>
                
                {/* Rows */}
                {words.map((word, rowIdx) => (
                  <div key={word} style={{ display: 'flex', marginBottom: '4px' }}>
                    {/* Word label */}
                    <div style={{ 
                      width: '120px', 
                      paddingRight: '10px',
                      textAlign: 'right',
                      fontWeight: '500',
                      fontSize: '0.9rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end'
                    }}>
                      {word}
                      {/* Add trend indicator */}
                      {(() => {
                        const firstHalf = matrix[rowIdx].slice(0, Math.floor(years.length/2));
                        const secondHalf = matrix[rowIdx].slice(Math.floor(years.length/2));
                        const firstAvg = firstHalf.reduce((a,b) => a+b, 0) / firstHalf.length;
                        const secondAvg = secondHalf.reduce((a,b) => a+b, 0) / secondHalf.length;
                        if (secondAvg > firstAvg * 1.5) {
                          return <TrendingUp size={16} color="#10b981" style={{ marginLeft: '4px' }} />;
                        } else if (secondAvg < firstAvg * 0.7) {
                          return <TrendingDown size={16} color="#ef4444" style={{ marginLeft: '4px' }} />;
                        }
                        return null;
                      })()}
                    </div>
                    
                    {/* Cells */}
                    {years.map((year, colIdx) => {
                      const value = matrix[rowIdx][colIdx];
                      return (
                        <div 
                          key={year}
                          style={{ 
                            width: '80px', 
                            height: '30px',
                            background: getColor(value, maxValue),
                            border: '1px solid #e2e8f0',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.75rem',
                            fontWeight: value > 0 ? '600' : '400',
                            color: value > maxValue * 0.5 ? 'white' : '#666',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                          }}
                          title={`${word} in ${year}: ${value} occurrences`}
                          onMouseEnter={(e) => {
                            e.target.style.transform = 'scale(1.1)';
                            e.target.style.zIndex = '10';
                            e.target.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
                          }}
                          onMouseLeave={(e) => {
                            e.target.style.transform = 'scale(1)';
                            e.target.style.zIndex = '1';
                            e.target.style.boxShadow = 'none';
                          }}
                        >
                          {value > 0 ? value : ''}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
              
              {/* Legend */}
              <div style={{ 
                marginTop: '1.5rem', 
                display: 'flex', 
                alignItems: 'center',
                gap: '1rem',
                fontSize: '0.85rem',
                color: '#666'
              }}>
                <span>Frequency:</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '20px', height: '20px', background: '#f3f4f6', border: '1px solid #e2e8f0' }}></div>
                  <span>None</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '20px', height: '20px', background: 'rgba(102, 126, 234, 0.3)', border: '1px solid #e2e8f0' }}></div>
                  <span>Low</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '20px', height: '20px', background: 'rgba(102, 126, 234, 0.6)', border: '1px solid #e2e8f0' }}></div>
                  <span>Medium</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '20px', height: '20px', background: 'rgba(102, 126, 234, 1)', border: '1px solid #e2e8f0' }}></div>
                  <span>High</span>
                </div>
              </div>
            </div>
          ) : (
            /* Timeline View - Year by Year */
            <div>
              {topicData.map(yearData => (
                <div key={yearData.year} style={{ 
                  marginBottom: '1.5rem', 
                  padding: '1rem', 
                  background: '#f8fafc', 
                  borderRadius: '8px' 
                }}>
                  <h4 style={{ color: '#667eea', marginBottom: '0.5rem' }}>
                    Year {yearData.year}
                  </h4>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {yearData.topics?.map((topic, idx) => (
                      <span key={idx} style={{
                        padding: '6px 12px',
                        background: 'white',
                        borderRadius: '20px',
                        fontSize: '14px',
                        border: '1px solid #e2e8f0',
                        fontWeight: idx < 3 ? 'bold' : 'normal',
                        color: idx < 3 ? '#667eea' : '#666'
                      }}>
                        {topic.word} ({topic.count})
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* Insights */}
          <div style={{ 
            marginTop: '1.5rem',
            padding: '1rem',
            background: '#fafafa',
            borderRadius: '8px',
            borderLeft: '4px solid #ff9800'
          }}>
            <strong>📊 What I Found:</strong>
            <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem' }}>
              {words[0]} appears most frequently across all years. 
              Topics with ↑ are emerging (grew {'>'}50% in recent years), 
              while ↓ indicates declining usage. The heatmap reveals temporal clusters - 
              notice how certain terms dominate specific time periods.
            </p>
          </div>
        </>
      ) : (
        <div className="empty-state-topic">
          <Brain size={48} />
          <p>
            Topic discovery uses advanced NLP to identify themes automatically.
            This feature requires processing - results will appear here once analysis is complete.
          </p>
        </div>
      )}
    </div>
  );
};

export default TopicDiscovery;