// components/TabNavigation.jsx
import React from 'react';

const TabNavigation = ({ tabs, activeTab, onTabChange }) => {
  return (
    <div className="tabs-container">
      <nav className="tabs-nav">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            title={tab.description || ''}  // Add tooltip
          >
            {tab.label}
          </button>
        ))}
      </nav>
      
      {/* Show description of active tab */}
      <div style={{ 
        padding: '0.5rem 1rem', 
        fontSize: '0.85rem', 
        color: '#666',
        fontStyle: 'italic',
        borderTop: '1px solid #f0f0f0',
        marginTop: '-1px'
      }}>
        {tabs.find(t => t.id === activeTab)?.description}
      </div>
    </div>
  );
};

export default TabNavigation;