// components/Header.jsx
import React from 'react';
import { Sparkles } from 'lucide-react';

const Header = () => {
  return (
    <header className="chronoscope-header">
      <div className="header-content">
        <div className="header-brand">
          <Sparkles size={32} />
          <div>
            <h1 className="header-title">The Brand Chronoscope</h1>
            <p className="header-subtitle">Track the Evolution of Brand Language Through Time</p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;