import * as tf from "@tensorflow/tfjs";
import { clearRect, displayPrediction, getCanvas, resetCanvas } from "./utils";

const clearButton = document.getElementById(
  "clear-button"
) as HTMLButtonElement;
const predictButton = document.getElementById(
  "check-button"
) as HTMLButtonElement;

clearButton.onclick = () => {
  resetCanvas();
  const predictionParagraph = document.getElementsByClassName("prediction")[0];
  predictionParagraph.textContent = "";
  clearRect();
};

let model: tf.LayersModel;

const modelPath = "./model/model.json";

const loadModel = async (path: string) => {
  if (!model) model = await tf.loadLayersModel(path);
};

const predict = async (img: HTMLImageElement) => {
  img.width = 200;
  img.height = 200;

  const processedImg = tf.browser.fromPixels(img, 4);
  const resizedImg = tf.image.resizeNearestNeighbor(processedImg, [28, 28]);

  const updatedImg = tf.cast(resizedImg, "float32");
  let shape;
  const predictions = await (
    model.predict(
      tf.reshape(updatedImg, (shape = [1, 28, 28, 4]))
    ) as tf.Tensor<tf.Rank>
  ).data();

  console.log(predictions);
  const label = predictions.indexOf(Math.max(...predictions));
  displayPrediction(label);
};

predictButton.onclick = async () => {
  const canvas = getCanvas();

  const drawing = canvas.toDataURL();
  const newImg = document.getElementsByClassName(
    "imageToCheck"
  )[0] as HTMLImageElement;

  newImg.src = drawing;

  newImg.onload = async () => {
    predict(newImg);
  };

  resetCanvas();
};

loadModel(modelPath);
