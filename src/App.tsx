import React from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const data = [
  { index: 0, value: 50 },
  { index: 1, value: 100 },
  { index: 2, value: 150 },
];

function App() {
  const surface = tfvis.visor().surface({ name: 'Barchart', tab: 'Charts' });

  return (
    <div className="App">

    </div>
  );
}

export default App;
