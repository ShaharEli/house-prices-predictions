import React, { useEffect, useState } from "react";
import {
  getNormalizedTensors,
  getPoints,
  HOUSES_CSV_URL,
  HOUSE_LABEL_NAME,
  loadData,
  plot,
} from "../utils";
import * as tf from "@tensorflow/tfjs";

const HousePricesAi = () => {
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  useEffect(() => {
    (async () => {
      const data = await loadData(HOUSES_CSV_URL);
      setAvailableFeatures(
        (await data.columnNames()).filter(
          (column: string) => column !== HOUSE_LABEL_NAME
        )
      );

      const points = await getPoints("sqft_living", data); //TODO: replace later to dynamic
      plot([points], ["sqft_living"], {
        xLabel: "sqft_living",
        yLabel: HOUSE_LABEL_NAME,
        name: `sqft_living vs price`,
      });

      const featureTensors = tf.tensor2d(
        points.map((p) => p.x),
        [points.length, 1]
      );
      const labelsTensors = tf.tensor2d(
        points.map((p) => p.y),
        [points.length, 1]
      );

      const {
        tensors: normalizedFeatureTensors,
        max: featureTensorsMax,
        min: featureTensorsMin,
      } = getNormalizedTensors(featureTensors);
      const {
        tensors: normalizedLabelTensors,
        max: labelTensorsMax,
        min: labelTensorsMin,
      } = getNormalizedTensors(labelsTensors);
    })();
  });
  return <div></div>;
};

export default HousePricesAi;
