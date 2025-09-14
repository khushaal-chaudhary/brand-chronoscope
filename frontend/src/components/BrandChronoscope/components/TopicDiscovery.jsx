// components/TopicDiscovery.jsx
import React from 'react';
import { Brain } from 'lucide-react';

const TopicDiscovery = ({ topicData }) => {
  return (
    <div style={{ 
    width: '100%', 
    //border: '2px solid blue',  // Debug border
    padding: '1rem',
    boxSizing: 'border-box'
    }}>
      <h3 className="section-title">AI-Powered Topic Discovery</h3>
      <p className="section-description">
        I let the algorithms find themes instead of searching for pre-defined keywords. 
        Turns out computers are better at finding patterns than humans - who knew? 
        This approach caught "metaverse" rising and falling faster than a cryptocurrency pump-and-dump.
      </p>
      
      {topicData && topicData.length > 0 ? (
        topicData.map(yearData => (
          <div key={yearData.year} className="topic-year-container">
            <h4 className="topic-year-title">Year {yearData.year}</h4>
            <div className="topic-pills">
              {yearData.topics && yearData.topics.map((topic, idx) => (
                <span key={idx} className="topic-pill">
                  {topic.word} ({topic.count})
                </span>
              ))}
            </div>
          </div>
        ))
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