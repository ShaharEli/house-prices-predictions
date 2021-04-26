import * as tf from "@tensorflow/tfjs";
import { TensorContainer } from "@tensorflow/tfjs";
import { IPlotProps, IPoint } from "../types";
import * as tfvis from "@tensorflow/tfjs-vis";
import { HOUSE_LABEL_NAME } from "./consts.utils";
export const loadData = async (dataUrl: string) => {
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
