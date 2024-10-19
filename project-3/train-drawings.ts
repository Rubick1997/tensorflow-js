import * as tf from "@tensorflow/tfjs-node-gpu";

import model from "./create-model";
import { loadData, getTrainData, getTestData } from "./get-data";

const train = async () => {
  loadData();
  const { images: trainImages, labels: trainLabels } = getTrainData();

  model.summary();

  await model.fit(trainImages, trainLabels, {
    epochs: 20,
    batchSize: 5,
    validationSplit: 0.5, // 20% of our training data will be used for validation
  });

  const { images: testImages, labels: testLabels } = getTestData();

  const evalOutput = model.evaluate(testImages, testLabels) as tf.Scalar[];

  const loss = evalOutput[0].dataSync()[0].toFixed(3);
  const accuracy = evalOutput[1].dataSync()[0].toFixed(3);

  console.log(`Loss: ${loss}, Accuracy: ${accuracy}`);

  await model.save("file://model");
};

train();
