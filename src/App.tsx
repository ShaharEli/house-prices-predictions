import React from "react";
import "./App.css";
import * as tfvis from "@tensorflow/tfjs-vis";
import HousePricesAi from "./pages/HousePricesAi";

const App = () => {
  return (
    <div className="App">
      <HousePricesAi />
    </div>
  );
};

export default App;
