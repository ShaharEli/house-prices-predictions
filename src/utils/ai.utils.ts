import * as tf from "@tensorflow/tfjs";
import { Tensor, Tensor, TensorContainer } from "@tensorflow/tfjs";
import {
  ILayer,
  INormalizedTensors,
  IPlotProps,
  IPoint,
  SetStatus,
} from "../types";
import * as tfvis from "@tensorflow/tfjs-vis";
import { HOUSE_LABEL_NAME } from "./consts.utils";
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
  points: [IPoint[]] | [IPoint[], IPoint[]],
  series: string[],
  { xLabel, yLabel, name }: IPlotProps
): void => {
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

interface ITrainModelOption {
  metrics: string[];
  name: string;
  batchSize: number;
  epochs: number;
  validationSplit: number;
}

const trainModel = (
  model: tf.Sequential,
  featureTensors: Tensor,
  labelsTensors: Tensor,
  opts: ITrainModelOption = {},
  setStatus: SetStatus,
  withVis: boolean
) => {};
