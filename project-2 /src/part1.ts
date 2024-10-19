declare const tmImage: {
  Webcam: new (width: number, height: number, fli: boolean) => any;
  load: (model: string, metadata: string) => any;
};

const path = "/src/my_model/";
const startButton = document.getElementById("start") as HTMLButtonElement;

let model: any;
let webCam: any;

const init = async () => {
  const modelPath = `${path}model.json`;
  const metadata = `${path}metadata.json`;

  model = await tmImage.load(modelPath, metadata);

  let maxPredictions = model.getTotalClasses();

  webCam = new tmImage.Webcam(200, 200, true);
  await webCam.setup();
  await webCam.play();
  window.requestAnimationFrame(loop);

  const webCamContainer = document.getElementById("webcam-container");
  if (!webCamContainer) return;
  webCamContainer.appendChild(webCam.canvas);
};

const predict = async () => {
  const predictions = await model.predict(webCam.canvas);

  const topPrediction = Math.max(...predictions.map((p: any) => p.probability));

  const topPredictionIndex = predictions.findIndex(
    (p: any) => p.probability === topPrediction
  );
  console.log(predictions[topPredictionIndex].className);
};

const loop = async () => {
  webCam.update();
  await predict();
  window.requestAnimationFrame(loop);
};

startButton.onclick = async () => await init();
