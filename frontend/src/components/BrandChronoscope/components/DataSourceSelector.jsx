// components/DataSourceSelector.jsx
import React, { useState } from 'react';
import { Database, FileSpreadsheet, Loader2, Info } from 'lucide-react';

const DataSourceSelector = ({
  availableDatasets,
  selectedDataset,
  uploadedFile,
  datasetInfo,
  isLoading,
  onDatasetSelect,
  onFileUpload
}) => {
  const [showUploadInfo, setShowUploadInfo] = useState(false);

  return (
    <div className="card">
      <h2 className="card-title">📊 Select Data Source</h2>
      
      <div className="data-sources">
        {availableDatasets.map(dataset => {
          const datasetKey = dataset.name.includes('Microsoft') ? 'microsoft' : 
                           dataset.name.includes('10-K') ? 'apple-10k' : 'apple-newsroom';
          
          return (
            <button
              key={dataset.name}
              onClick={() => onDatasetSelect(dataset.name)}
              className={`data-source-btn ${selectedDataset === datasetKey ? 'selected' : ''}`}
            >
              <Database size={20} />
              {dataset.name}
            </button>
          );
        })}
        
        <div style={{ position: 'relative' }}>
          <label 
            htmlFor="file-upload"
            className={`upload-label ${uploadedFile ? 'uploaded' : ''}`}
            onMouseEnter={() => setShowUploadInfo(true)}
            onMouseLeave={() => setShowUploadInfo(false)}
          >
            <input
              type="file"
              accept=".csv"
              onChange={onFileUpload}
              className="upload-input"
              id="file-upload"
            />
            <FileSpreadsheet size={20} />
            {uploadedFile ? uploadedFile.name : 'Upload CSV'}
            <Info size={16} style={{ marginLeft: '4px', opacity: 0.7 }} />
          </label>
          
          {showUploadInfo && (
            <div style={{
              position: 'absolute',
              bottom: '100%',
              left: '50%',
              transform: 'translateX(-50%)',
              marginBottom: '8px',
              padding: '12px',
              background: 'white',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
              width: '300px',
              zIndex: 1000
            }}>
              <strong style={{ display: 'block', marginBottom: '8px', color: '#2c3e50' }}>
                CSV Format Requirements:
              </strong>
              <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.85rem', color: '#555' }}>
                <li><strong>year</strong> column (e.g., 2015, 2016...)</li>
                <li><strong>text</strong> column with document content</li>
                <li>Optional: <strong>source</strong> column</li>
              </ul>
              <div style={{ marginTop: '8px', fontSize: '0.8rem', color: '#666', fontStyle: 'italic' }}>
                Pro tip: Each row should be one document (annual report, press release, etc.) 
                with at least 100 words for meaningful analysis.
              </div>
            </div>
          )}
        </div>
      </div>

      {datasetInfo && (
        <div className="dataset-info">
          <p>
            Loaded: <strong>{datasetInfo.rows}</strong> documents | 
            Years: <strong>{datasetInfo.years?.[0]}-{datasetInfo.years?.[datasetInfo.years.length - 1]}</strong>
          </p>
        </div>
      )}

      {isLoading && (
        <div className="loading-state">
            <Loader2 size={20} className="loading-spinner" />
            <span>Crunching numbers... Finding patterns... Questioning life choices...</span>
        </div>
      )}
    </div>
  );
};

export default DataSourceSelector;