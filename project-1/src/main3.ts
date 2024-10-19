import "@tensorflow/tfjs";
// import "@mediapipe/face_detection";
import "@tensorflow/tfjs-core";
import * as faceDetection from "@tensorflow-models/face-detection";
import { drawFaceBox, startWebcam, takePicture } from "../utils";

const webcamButton = document.getElementById("webcam") as HTMLButtonElement;
const captureButton = document.getElementById("capture") as HTMLButtonElement;
const video = document.querySelector("video") as HTMLVideoElement;

let model: faceDetection.SupportedModels.MediaPipeFaceDetector;
let detector: faceDetection.FaceDetector;

const init = async () => {
  model = faceDetection.SupportedModels.MediaPipeFaceDetector;
  detector = await faceDetection.createDetector(model, { runtime: "tfjs" });
};

const predict = async (photo: HTMLCanvasElement) => {
  const faces = await detector.estimateFaces(photo, { flipHorizontal: false });

  drawFaceBox(photo, faces);
};

webcamButton.onclick = () => startWebcam(video);
captureButton.onclick = () => takePicture(video, predict);

init();
