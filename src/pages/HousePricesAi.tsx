import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  createModel,
  getNormalizedTensors,
  getPoints,
  HOUSES_CSV_URL,
  HOUSE_LABEL_NAME,
  LABELS_TO_IGNORE,
  loadData,
  modelStatusEnum,
  plot,
  plotPredictionLine,
  predict,
  testModelCb,
  trainModelCb,
} from "../utils";
import * as tf from "@tensorflow/tfjs";
import { Button, Input } from "@material-ui/core";
import { IMetaData, IPoint } from "../types";

let data: tf.data.CSVDataset | null;
let points: IPoint[] | null;
const HousePricesAi = () => {
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [model, setModel] = useState<tf.Sequential | null>();
  const [currStatus, setCurrStatus] = useState<keyof typeof modelStatusEnum>(0);
  const [metaData, setMetaData] = useState<IMetaData | null>();
  const [predictionInput, setPredictionInput] = useState<string>("");
  const [predictedVal, setPredictedVal] = useState<string | number>("");
  const [testVal, setTestVal] = useState<string>("");

  const prepareData = useCallback(async () => {
    if (!data) return;
    setStatus("creating points");
    points = await getPoints(selectedFeature, data); //TODO: replace later to dynamic
    plot([points], [selectedFeature], {
      xLabel: selectedFeature,
      yLabel: HOUSE_LABEL_NAME,
      name: `${selectedFeature} vs price`,
    });
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
    setModel(createModel(setStatus, true));
    setCurrStatus(1);
    setMetaData({
      trainingLabels,
      testingLabels,
      trainingFeatures,
      testingFeatures,
      labelTensorsMax,
      labelTensorsMin,
      featureTensorsMax,
      featureTensorsMin,
    });
    setStatus("finished creating model");
  }, [selectedFeature]);

  const predictDisabled = useMemo(() => ![2, 3].includes(currStatus), [
    currStatus,
  ]);

  const predictVal = async () => {
    if (!metaData || !model) return setPredictedVal("");
    const predictedValue = await predict(
      predictionInput,
      metaData.featureTensorsMax,
      metaData.featureTensorsMin,
      metaData.labelTensorsMin,
      metaData.labelTensorsMax,
      model
    );
    if (!predictedValue || typeof predictedValue !== "string")
      return setPredictedVal("");
    setPredictedVal(predictedValue);
  };

  const testModel = () => {
    if (!metaData || !model) return;
    const testResult = testModelCb(
      model,
      metaData.testingFeatures,
      metaData.testingLabels
    );
    setTestVal(testResult);
  };

  const trainModel = useCallback(async () => {
    if (!metaData || !model || !points) return;
    const history = await trainModelCb(
      model,
      metaData.trainingFeatures,
      metaData.trainingLabels,
      { name: "loss", metrics: ["loss"], validationSplit: 0.1 },
      setStatus,
      metaData.featureTensorsMax,
      metaData.featureTensorsMin,
      metaData.labelTensorsMin,
      metaData.labelTensorsMax,
      points,
      selectedFeature,
      {
        xLabel: selectedFeature,
        yLabel: HOUSE_LABEL_NAME,
        name: `${selectedFeature} vs price`,
      }
    );

    // console.log(history);
  }, [points, metaData]);

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
          <Button disabled={currStatus !== 1} onClick={trainModel}>
            train model
          </Button>
          <br />
          <Button disabled={currStatus < 2 || !!testVal} onClick={testModel}>
            test model
          </Button>
          <br />
          {testVal && (
            <>
              <h3>test result: {testVal}</h3>
              <br />
            </>
          )}
          <Input
            value={predictionInput}
            onChange={({ target: { value } }) => setPredictionInput(value)}
            disabled={predictDisabled}
            placeholder={`Enter ${selectedFeature} amount to get predicted ${HOUSE_LABEL_NAME}`}
          />
          <Button
            disabled={predictDisabled || !predictionInput}
            onClick={predictVal}
          >
            predict
          </Button>
          {predictedVal && <p>predicted: {predictedVal}</p>}
        </div>
      )}
    </div>
  );
};

export default HousePricesAi;
