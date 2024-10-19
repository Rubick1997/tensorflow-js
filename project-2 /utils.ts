// Part 1
// -----------

import { DetectedObject } from "@tensorflow-models/coco-ssd";
import { Face } from "@tensorflow-models/face-detection";

export const showResult = (classes: DetectedObject[]) => {
  const predictionsElement = document.getElementById("predictions");
  const probsContainer = document.createElement("div");
  for (let i = 0; i < classes.length; i++) {
    probsContainer.innerText = `Prediction: ${classes[i].class}, Probability: ${classes[i].score}`;
  }
  if (!predictionsElement) return;
  predictionsElement.appendChild(probsContainer);
};

export const IMAGE_SIZE = 224;

export const handleFilePicker = (callback: (img: HTMLImageElement) => void) => {
  const fileElement = document.getElementById("file");
  if (!fileElement) return;
  fileElement.addEventListener("change", (evt: Event) => {
    if (!evt.target) return;
    let file = (evt.target as HTMLInputElement).files as FileList;
    let f = file[0];

    if (!f.type.match("image.*")) {
      return;
    }

    let reader = new FileReader();
    reader.onload = (e) => {
      let img = document.createElement("img");
      if (!e.target) return;
      img.src = e.target.result as string;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      const loadedImgElement = document.getElementById("loaded-image");
      if (!loadedImgElement) return;

      loadedImgElement.appendChild(img);

      img.onload = () => callback(img);

      // img.onload = () => predict(img);
    };
    reader.readAsDataURL(f);
  });
};

// Part 2
// -----------

export const startWebcam = (video: HTMLVideoElement) => {
  return navigator.mediaDevices
    .getUserMedia({
      audio: false,
      video: { width: 320, height: 185 },
    })
    .then((stream) => {
      video.srcObject = stream;
      // track = stream.getTracks()[0];
      video.onloadedmetadata = () => video.play();
    })
    .catch((err) => {
      /* handle the error */
    });
};

export const takePicture = (
  video: HTMLVideoElement,
  callback: (canvas: HTMLCanvasElement) => Promise<void>
): void => {
  const predictButton = document.getElementById("predict") as HTMLButtonElement;
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const width = IMAGE_SIZE; // We will scale the photo width to this
  const height = IMAGE_SIZE;
  const context = canvas.getContext("2d");

  if (!context) {
    console.error("Failed to get 2D context");
    return;
  }

  canvas.width = width;
  canvas.height = height;
  context.drawImage(video, 0, 0, width, height);

  const outputEl = document.getElementById("predictions");
  if (outputEl) {
    outputEl.appendChild(canvas);
  }

  if (predictButton) {
    predictButton.disabled = false;
    predictButton.onclick = async () => {
      await callback(canvas);
    };
  }
};

// Part 3
// -----------

export const drawFaceBox = (photo: HTMLCanvasElement, faces: Face[]) => {
  // Draw box around the face detected ⬇️
  // ------------------------------------
  const faceCanvas = document.createElement("canvas");
  faceCanvas.width = IMAGE_SIZE;
  faceCanvas.height = IMAGE_SIZE;
  faceCanvas.style.position = "absolute";
  faceCanvas.style.left = photo.offsetLeft.toString();
  faceCanvas.style.top = photo.offsetTop.toString();
  const ctx = faceCanvas.getContext("2d");
  if (!ctx) return;
  ctx.beginPath();
  ctx.strokeStyle = "red";
  ctx.strokeRect(
    faces[0].box.xMin,
    faces[0].box.yMin,
    faces[0].box.width,
    faces[0].box.height
  );

  const webcamSection = document.getElementById("webcam-section");
  if (!webcamSection) return;
  webcamSection.appendChild(faceCanvas);
};
