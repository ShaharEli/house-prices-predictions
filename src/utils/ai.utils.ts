import * as tf from "@tensorflow/tfjs";
import { Tensor, History, TensorContainer } from "@tensorflow/tfjs";
import {
  ILayer,
  INormalizedTensors,
  IPlotProps,
  IPoint,
  SetStatus,
} from "../types";
import * as tfvis from "@tensorflow/tfjs-vis";
import { HOUSE_LABEL_NAME, storageID } from "./consts.utils";
export const loadData = async (
  dataUrl: string
): Promise<tf.data.CSVDataset> => {
  await tf.ready();
  return tf.data.csv(dataUrl);
};

export const getPoints = async (
  featureToTest: string,
  data: tf.data.CSVDataset
): Promise<IPoint[]> => {
  const points = await data
    .map((p: TensorContainer) => ({
      // @ts-ignore
      x: p[featureToTest],
      // @ts-ignore
      y: p[HOUSE_LABEL_NAME],
    }))
    .toArray();

  if (points.length % 2 !== 0) {
    points.pop(); //making the points dividable
  }
  tf.util.shuffle(points); //randomizing the points to avoid overfitting
  return points;
};

export const plot = (
  points: [IPoint[]],
  series: string[],
  { xLabel, yLabel, name }: IPlotProps,
  predictedPoints?: IPoint[]
): void => {
  if (predictedPoints) {
    //   @ts-ignore
    points.push(predictedPoints);
    series.push("predicted");
  }
  const data = { values: points, series };
  tfvis.render.scatterplot({ name }, data, { xLabel, yLabel });
};
export const getNormalizedTensors = (
  preNormalizedTensors: Tensor,
  prevMax?: number,
  prevMin?: number
): INormalizedTensors => {
  let max, min;
  if (prevMax !== undefined && prevMin !== undefined) {
    max = prevMax;
    min = prevMin;
  } else {
    max = preNormalizedTensors.max().dataSync()[0];
    min = preNormalizedTensors.min().dataSync()[0];
  }
  const tensors = preNormalizedTensors.sub(min).div(max - min);
  return {
    tensors,
    max,
    min,
  };
};

export const createModel = (
  setStatus: SetStatus,
  showVisual?: boolean
): tf.Sequential => {
  setStatus("preparing layers");

  const layers: ILayer[] = [
    {
      activation: "sigmoid",
      units: 8,
      useBias: true,
      inputDim: 1,
    },
    {
      activation: "sigmoid",
      units: 1,
      useBias: true,
    },
  ];
  const model = tf.sequential({
    //   @ts-ignore
    layers: layers.map((layer) => tf.layers.dense(layer)),
  });
  setStatus("preparing optimizer");

  const optimizer = tf.train.adam();
  setStatus("compiling model");

  model.compile({
    optimizer,
    loss: "meanSquaredError",
  });
  setStatus("finished compiling model");
  if (showVisual) {
    // Showing model
    tfvis.show.modelSummary({ name: "model" }, model);
    // Showing the first layer
    tfvis.show.layer({ name: "model" }, model.getLayer(undefined, 0));
  }

  return model;
};
const denormalise = (tensor: any, max: number, min: number): Tensor =>
  tensor.mul(max - min).add(min);

export const predict = (
  input: string,
  featureMax: number,
  featureMin: number,
  labelMin: number,
  labelMax: number,
  model: tf.Sequential
): number | boolean => {
  const parsedInput = parseInt(input);
  if (isNaN(parsedInput)) {
    return false;
  } else {
    return tf.tidy(() => {
      const inputTensor = tf.tensor1d([parsedInput]);

      const normalisedInput = getNormalizedTensors(
        inputTensor,
        featureMax,
        featureMin
      );
      const normalisedOutputTensor = model.predict(normalisedInput.tensors);
      // @ts-ignore

      const outputTensor = denormalise(
        normalisedOutputTensor,
        labelMax,
        labelMin
      );
      const outputValue = outputTensor.dataSync()[0];
      return outputValue;
    });
  }
};

export const plotPredictionLine = (
  model: tf.Sequential,
  featureMax: number,
  featureMin: number,
  labelMin: number,
  labelMax: number,
  points: IPoint[],
  selectedFeature: string,
  { xLabel, yLabel, name }: IPlotProps
) => {
  const [xs, ys] = tf.tidy(() => {
    const normalisedXs = tf.linspace(0, 1, 100);
    const normalisedYs = model.predict(normalisedXs.reshape([100, 1]));

    const xs = denormalise(normalisedXs, featureMax, featureMin);
    const ys = denormalise(normalisedYs, labelMax, labelMin);

    return [xs.dataSync(), ys.dataSync()];
  });

  const predictedPoints: IPoint[] = Array.from(xs).map((val, i) => {
    return { x: val, y: ys[i] };
  });

  plot(
    [points],
    [selectedFeature],
    {
      xLabel,
      yLabel,
      name,
    },
    predictedPoints
  );
};

interface ITrainModelOption {
  metrics?: string[];
  name: string;
  batchSize?: number;
  epochs?: number;
  validationSplit?: number;
  shuffle?: boolean;
}

export const testModelCb = (
  model: tf.Sequential,
  testingFeatures: Tensor,
  testingLabels: Tensor
) => {
  const evaluation = model.evaluate(testingFeatures, testingLabels);
  //   @ts-ignore
  return evaluation.dataSync()[0];
};

export const saveModel = async (featureName: string, model: tf.Sequential) =>
  await model.save(`localstorage://${storageID}-${featureName}`);

export const loadModel = async (featureName: string) => {
  const storageKey = `localstorage://${storageID}-${featureName}`;
  const models = await tf.io.listModels();
  const modelInfo = models[storageKey];
  if (modelInfo) {
    const prevModel = await tf.loadLayersModel(storageKey);
    return prevModel;
  } else {
    return null;
  }
};

export const trainModelCb = (
  model: tf.Sequential,
  featureTensors: Tensor,
  labelsTensors: Tensor,
  opts: ITrainModelOption,
  setStatus: SetStatus,
  featureMax: number,
  featureMin: number,
  labelMin: number,
  labelMax: number,
  points: IPoint[],
  selectedFeature: string,
  { xLabel, yLabel, name }: IPlotProps
): Promise<History> => {
  const metrics = opts.metrics || ["loss"];

  const container = {
    name: opts.name,
  };
  setStatus("creating callbacks");
  const callbacks = tfvis.show.fitCallbacks(container, metrics);

  const trainingOpts = {
    batchSize: opts.batchSize || 50, //50
    epochs: opts.epochs || 50, //50
    validationSplit: opts.validationSplit || null,
    callbacks: {
      //   onEpochEnd: callbacks.onEpochEnd,
      //   onBatchEnd: callbacks.onBatchEnd,
      onEpochEnd: () => {
        plotPredictionLine(
          model,
          featureMax,
          featureMin,
          labelMin,
          labelMax,
          points,
          selectedFeature,
          { xLabel, yLabel, name }
        );
        const layer = model.getLayer(undefined, 0);
        tfvis.show.layer({ name: "Layer 1" }, layer);
      },
    },
    shuffle: opts.shuffle || true,
  };
  setStatus("training");
  //   @ts-ignore
  return model.fit(featureTensors, labelsTensors, trainingOpts);
};
