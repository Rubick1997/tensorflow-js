import "@tensorflow/tfjs";
import * as cocSsd from "@tensorflow-models/coco-ssd";
import { showResult, startWebcam, takePicture } from "../utils";

const webcamButton = document.getElementById("webcam") as HTMLButtonElement;
const captureButton = document.getElementById("capture") as HTMLButtonElement;
const video = document.querySelector("video") as HTMLVideoElement;

let model: cocSsd.ObjectDetection;

const init = async () => {
  model = await cocSsd.load();
};

const predict = async (canvas: HTMLCanvasElement) => {
  const predictions = await model.detect(canvas);
  console.log(predictions);
  showResult(predictions);
};

webcamButton.onclick = () => startWebcam(video);
captureButton.onclick = () => takePicture(video, predict);

init();
