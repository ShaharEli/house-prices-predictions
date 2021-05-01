import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  createModel,
  getNormalizedTensors,
  getPoints,
  HOUSES_CSV_URL,
  HOUSE_LABEL_NAME,
  LABELS_TO_IGNORE,
  loadData,
  loadModel,
  modelStatusEnum,
  plot,
  predict,
  saveModel,
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
  const [prevModel, setPrevModel] = useState<tf.Sequential | null>();
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
    if (model) {
      setStatus("data created");
      setCurrStatus(3);
      return null;
    }
    setStatus("creating model");
    const newModel = createModel(setStatus, true);
    setModel(newModel);
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
    return {
      trainingLabels,
      testingLabels,
      trainingFeatures,
      testingFeatures,
      labelTensorsMax,
      labelTensorsMin,
      featureTensorsMax,
      featureTensorsMin,
      newModel,
    };
  }, [selectedFeature.length, model]);

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
    if (!predictedValue) return setPredictedVal("invalid input");
    setPredictedVal(Number(predictedValue).toLocaleString());
    setPredictionInput("");
  };

  const loadModelCB = async () => {
    setModel(prevModel);
    setPrevModel(null);
  };

  const testModel = async () => {
    if (!metaData || !model) return;
    const testResult = await testModelCb(
      model,
      metaData.testingFeatures,
      metaData.testingLabels
    );

    setTestVal(testResult);
  };

  const saveModelCb = async () => {
    if (!model) return;
    setStatus("saving");
    setPrevModel(null);
    await saveModel(selectedFeature, model);
    setStatus("model saved");
    setCurrStatus(3);
  };

  const trainModel = useCallback(async () => {
    const modelMetaData = await prepareData();
    if (!modelMetaData || !points) return;
    const {
      trainingLabels,
      trainingFeatures,
      labelTensorsMax,
      labelTensorsMin,
      featureTensorsMax,
      featureTensorsMin,
      newModel,
    } = modelMetaData;
    if (!points) return;
    const history = await trainModelCb(
      newModel,
      trainingFeatures,
      trainingLabels,
      { name: "loss", metrics: ["loss"], validationSplit: 0.1 },
      setStatus,
      featureTensorsMax,
      featureTensorsMin,
      labelTensorsMin,
      labelTensorsMax,
      points,
      selectedFeature,
      {
        xLabel: selectedFeature,
        yLabel: HOUSE_LABEL_NAME,
        name: `${selectedFeature} vs price`,
      }
    );
    setCurrStatus(2);
    const {
      history: { loss: trainningLoss },
    } = history;
    const loss = trainningLoss[trainningLoss.length - 1];
    if (typeof loss === "number")
      setStatus(`Model trained with ${loss.toPrecision(6)} loss`);
  }, [points, metaData, selectedFeature]);

  useEffect(() => {
    if (selectedFeature) {
      (async () => {
        const prevModel = await loadModel(selectedFeature);
        if (prevModel) {
          // @ts-ignore
          setPrevModel(prevModel);
        }
      })();
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
          <Button
            disabled={
              currStatus !== 0 ||
              status === "training" ||
              status === "creating points"
            }
            onClick={trainModel}
          >
            train model
          </Button>
          <br />
          <Button disabled={currStatus < 2 || !!testVal} onClick={testModel}>
            test model
          </Button>
          <br />
          {testVal && (
            <>
              <h3>test result: {Number(testVal).toPrecision(6)} loss</h3>
              <br />
            </>
          )}

          <Button
            disabled={currStatus !== 2 || status === "saving"}
            onClick={saveModelCb}
          >
            save model
          </Button>
          <br />

          <Button disabled={!prevModel} onClick={loadModelCB}>
            load model
          </Button>
          <br />
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
