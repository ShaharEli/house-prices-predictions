import { Tensor } from "@tensorflow/tfjs";

import { ActivationSerialization } from "@tensorflow/tfjs-layers/src/keras_format/activation_config";
export interface IPoint {
  x: number;
  y: number;
}

export interface IPlotProps {
  xLabel: string;
  yLabel: string;
  name: string;
}

export interface INormalizedTensors {
  tensors: Tensor;
  max: number;
  min: number;
}

export interface ILayer {
  activation: ActivationSerialization;
  units: number;
  useBias: boolean;
  inputDim?: number;
}

export type SetStatus = (arg: string) => void;

export interface ICallbacks {
  onEpochEnd: () => void;
  onBatchEnd: () => void;
}
export interface IMetaData {
  featureTensorsMax: number;
  featureTensorsMin: number;
  labelTensorsMax: number;
  labelTensorsMin: number;
  trainingFeatures: Tensor;
  testingFeatures: Tensor;
  trainingLabels: Tensor;
  testingLabels: Tensor;
}
