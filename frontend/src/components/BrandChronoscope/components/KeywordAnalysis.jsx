// components/KeywordAnalysis.jsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ArrowUp, ArrowDown } from 'lucide-react';

const KeywordAnalysis = ({ keywordData, emergingTerms, decliningTerms }) => {
  // Color for each line
  const getLineColor = (index) => {
    const colors = ['#667eea', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899'];
    return colors[index % colors.length];
  };

  return (
    <div style={{ 
    width: '100%', 
    //border: '2px solid red',  // Debug border
    padding: '1rem',
    boxSizing: 'border-box'
    }}>
      <h3 className="section-title">Keyword Frequency Evolution</h3>
      
      <div style={{ width: '100%', height: 500, marginBottom: '2rem' }}>
        <ResponsiveContainer>
          <LineChart 
            data={keywordData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="year" stroke="#64748b" />
            <YAxis stroke="#64748b" />
            <Tooltip />
            <Legend />
            {Object.keys(keywordData[0] || {})
              .filter(k => k !== 'year')
              .map((keyword, index) => (
                <Line
                  key={keyword}
                  type="monotone"
                  dataKey={keyword}
                  stroke={getLineColor(index)}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="terms-grid">
        <div className="terms-section">
          <h4 className="emerging-header">
            <ArrowUp size={20} />
            Rising Terms
          </h4>
          {emergingTerms.length > 0 ? (
            emergingTerms.map((term, idx) => (
              <div key={idx} className="term-card emerging">
                <div className="term-info">
                  <div className="term-name">{term.keyword}</div>
                  <div className="term-frequency">
                    Current: {term.recent_frequency} per 10k words
                  </div>
                </div>
                <div className="term-change">
                  +{term.growth_rate}%
                </div>
              </div>
            ))
          ) : (
            <p style={{ color: '#64748b' }}>No emerging terms detected</p>
          )}
        </div>

        <div className="terms-section">
          <h4 className="declining-header">
            <ArrowDown size={20} />
            Declining Terms
          </h4>
          {decliningTerms.length > 0 ? (
            decliningTerms.map((term, idx) => (
              <div key={idx} className="term-card declining">
                <div className="term-info">
                  <div className="term-name">{term.keyword}</div>
                  <div className="term-frequency">
                    Current: {term.recent_frequency} per 10k words
                  </div>
                </div>
                <div className="term-change">
                  {term.decline_rate}%
                </div>
              </div>
            ))
          ) : (
            <p style={{ color: '#64748b' }}>No declining terms detected</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default KeywordAnalysis;