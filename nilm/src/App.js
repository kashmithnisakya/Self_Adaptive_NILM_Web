import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Import the CSS file

function App() {
  const [deviceName, setDeviceName] = useState('');
  const [year, setYear] = useState('');
  const [startPoint, setStartPoint] = useState('');
  const [endPoint, setEndPoint] = useState('');
  const [response, setResponse] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const deviceData = {
      device_name: deviceName,
      year: parseInt(year),
      start_point: parseInt(startPoint),
      end_point: parseInt(endPoint),
    };
    try {
      const res = await axios.post('http://localhost:8000/predict/', deviceData);
      setResponse(res.data);
      console.log(res.data);
    } catch (error) {
      console.error('There was an error!', error);
    }
  };

  return (
    <div className="App">
      <h1>Device Prediction Form</h1>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Device Name:</label>
          <input 
            type="text" 
            value={deviceName} 
            onChange={(e) => setDeviceName(e.target.value)} 
            required 
          />
        </div>
        <div className="form-group">
          <label>Year:</label>
          <input 
            type="number" 
            value={year} 
            onChange={(e) => setYear(e.target.value)} 
            required 
          />
        </div>
        <div className="form-group">
          <label>Start Point:</label>
          <input 
            type="number" 
            value={startPoint} 
            onChange={(e) => setStartPoint(e.target.value)} 
            required 
          />
        </div>
        <div className="form-group">
          <label>End Point:</label>
          <input 
            type="number" 
            value={endPoint} 
            onChange={(e) => setEndPoint(e.target.value)} 
            required 
          />
        </div>
        <button type="submit">Submit</button>
      </form>
      {response && (
        <div className="response">
          <h2>Response from the server:</h2>
          <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default App;
