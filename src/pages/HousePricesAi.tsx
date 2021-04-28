import React, { useCallback, useEffect, useState } from "react";
import {
  createModel,
  getNormalizedTensors,
  getPoints,
  HOUSES_CSV_URL,
  HOUSE_LABEL_NAME,
  LABELS_TO_IGNORE,
  loadData,
  plot,
} from "../utils";
import * as tf from "@tensorflow/tfjs";
import { Button } from "@material-ui/core";

let data: tf.data.CSVDataset | null;
const HousePricesAi = () => {
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<string>("");
  const [status, setStatus] = useState<string>("");

  const prepareData = useCallback(async () => {
    if (!data) return;
    setStatus("creating points");
    const points = await getPoints(selectedFeature, data); //TODO: replace later to dynamic
    plot([points], [selectedFeature], {
      xLabel: selectedFeature,
      yLabel: HOUSE_LABEL_NAME,
      name: `${selectedFeature} vs price`,
    });
    tf.tidy(() => {
      setStatus("creating tensors");
      const featureTensors = tf.tensor2d(
        points.map((p) => p.x),
        [points.length, 1]
      );
      const labelsTensors = tf.tensor2d(
        points.map((p) => p.y),
        [points.length, 1]
      );
      setStatus("normalizing tensors");
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
      const eightyPercentOfThePoints = Math.floor(points.length * 0.8);
      const twentyPercentOfThePoints = points.length - eightyPercentOfThePoints;
      const arrToSplitBy = [eightyPercentOfThePoints, twentyPercentOfThePoints];
      setStatus("splitting tensors to train and test");
      const [trainingFeatures, testingFeatures] = tf.split(
        normalizedFeatureTensors,
        arrToSplitBy
      );
      const [trainingLabels, testingLabels] = tf.split(
        normalizedLabelTensors,
        arrToSplitBy
      );
      setStatus("creating model");
      const model = createModel(setStatus, true);
    });
  }, [selectedFeature]);

  useEffect(() => {
    if (selectedFeature) {
      prepareData();
    }
  }, [selectedFeature]);

  useEffect(() => {
    (async () => {
      data = await loadData(HOUSES_CSV_URL);
      setAvailableFeatures(
        (await data.columnNames()).filter(
          (column: string) => !LABELS_TO_IGNORE.includes(column)
        )
      );
    })();
  });
  return (
    <div>
      {!selectedFeature ? (
        <div>
          <h2>select the feature you want to train</h2>
          {availableFeatures.map((feature) => (
            <Button key={feature} onClick={() => setSelectedFeature(feature)}>
              {feature}{" "}
            </Button>
          ))}
        </div>
      ) : (
        <div>
          <h2>selected feature: {selectedFeature}</h2>
          <h3>status: {status}</h3>
        </div>
      )}
    </div>
  );
};

export default HousePricesAi;
