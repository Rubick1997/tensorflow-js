import "./styles.css";
import "@tensorflow/tfjs";
import * as cocSsd from "@tensorflow-models/coco-ssd";
import { handleFilePicker, showResult } from "../utils";

let model: cocSsd.ObjectDetection;

const predict = async (img: HTMLImageElement) => {
  const predictions = await model.detect(img);
  console.log(predictions);
  showResult(predictions);
};

const init = async () => {
  model = await cocSsd.load();

  handleFilePicker(predict);
};

init();
