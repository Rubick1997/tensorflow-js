import * as tf from "@tensorflow/tfjs";
import * as tfd from "@tensorflow/tfjs-data";
import { WebcamIterator } from "@tensorflow/tfjs-data/dist/iterators/webcam_iterator";

const recordButtons = document.getElementsByClassName("record-button");
const buttonsContainer = document.getElementById(
  "buttons-container"
) as HTMLDivElement;

const trainButton = document.getElementById("train") as HTMLButtonElement;
const predictButton = document.getElementById("predict") as HTMLButtonElement;
const statusElement = document.getElementById("status") as HTMLDivElement;

let webCam: WebcamIterator,
  initialModel: tf.LayersModel,
  mouseDown: boolean,
  newModel: tf.Sequential;

const totals = [0, 0];
const labels = ["left", "right"];

// when tuning these parameters it is more art than science
const learningRate = 0.0001; // how frequently the models change during training
const batchSizeFraction = 0.4; // number of training examples used in each iteration
const epochs = 30; // steps to train teh model

const denseUnits = 100; // number of outputs of the layer

let isTraining = false;
let isPredicting = false;

// loading the initial modal and creating new model that we can work with when we will be creating the new model
const loadModel = async () => {
  // extracting different layers of the ML model
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
  );
  const layer = mobilenet.getLayer("conv_pw_13_relu");
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
};

const getImage = async () => {
  const img = await webCam.capture();
  const processedImg = tf.tidy(() => {
    return img.expandDims(0).toFloat().div(127).sub(1);
  });
  img.dispose();
  return processedImg;
};

let xs: tf.Tensor<tf.Rank> | null = null; //actual examples data
let xy: tf.Tensor<tf.Rank> | null = null; // labels attached to the data

const addExample = async (index: number) => {
  let img = await getImage();
  let example = initialModel.predict(img) as tf.Tensor<tf.Rank>;

  const tensor = tf.tensor1d([index]).toInt();

  const y = tf.tidy(() => tf.oneHot(tensor, labels.length));

  if (xs === null || xy === null) {
    xs = tf.keep(example);
    xy = tf.keep(y);
  } else {
    const previousX = xs;
    xs = tf.keep(previousX.concat(example, 0));

    const previousY = xy;
    xy = tf.keep(previousY.concat(y, 0));

    previousX.dispose();
    previousY.dispose();
    y.dispose();
    img.dispose;
  }
};

const handleAddExample = async (labelIndex: number) => {
  mouseDown = true;
  const total = document.getElementById(
    `${labels[labelIndex]}-total`
  ) as HTMLSpanElement;

  while (mouseDown) {
    addExample(labelIndex);
    total.innerText = (++totals[labelIndex]).toString();

    await tf.nextFrame();
  }
};

const init = async () => {
  webCam = await tfd.webcam(
    document.getElementById("webcam") as HTMLVideoElement
  );
  initialModel = await loadModel();
  statusElement.style.display = "none";
  (document.getElementById("controller") as HTMLDivElement).style.display =
    "block";
};

const train = async () => {
  isTraining = true;
  if (!xs || !xy) {
    throw new Error("You forgot to add examples before training");
  }

  newModel = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: initialModel.outputs[0].shape.slice(1) }),
      tf.layers.dense({
        units: denseUnits,
        activation: "relu", // rectified linear unit activation function
        kernelInitializer: "varianceScaling", // initializing the weights of the layer
        useBias: true,
      }),
      tf.layers.dense({
        units: labels.length,
        kernelInitializer: "varianceScaling",
        useBias: true,
        activation: "softmax",
      }),
    ],
  });

  const optimizer = tf.train.adam(learningRate);
  newModel.compile({ optimizer, loss: "categoricalCrossentropy" });

  const batchSize = Math.floor(xs.shape[0] * batchSizeFraction);

  newModel.fit(xs, xy, {
    batchSize,
    epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        statusElement.innerHTML = `Loss: ${logs?.loss.toFixed(5)}`;
      },
    },
  });

  isTraining = false;
};

init();

buttonsContainer.onmousedown = (e) => {
  if (e.target === recordButtons[0]) {
    handleAddExample(0);
  } else {
    handleAddExample(1);
  }
};

buttonsContainer.onmouseup = () => {
  mouseDown = false;
};

trainButton.onclick = async () => {
  train();
  statusElement.style.display = "block";
  statusElement.innerHTML = "Training...";
};

predictButton.onclick = async () => {
  isPredicting = true;
  while (isPredicting) {
    const img = await getImage();

    const initialModelPrediction = initialModel.predict(img);
    const predictions = newModel.predict(initialModelPrediction) as tf.Tensor;

    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    console.log(labels[classId]);

    img.dispose();
    await tf.nextFrame();
  }
};
