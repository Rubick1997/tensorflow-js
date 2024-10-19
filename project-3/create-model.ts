import * as tf from "@tensorflow/tfjs";

// kernel is a way to transform the data from one layer and pass it to the next
const kernelSize = [3, 3];
const filter = 32;
const numClasses = 2;

const model = tf.sequential();

model.add(
  tf.layers.conv2d({
    inputShape: [28, 28, 4],
    kernelSize,
    filters: filter,
    activation: "relu",
  })
);

model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({ units: 10, activation: "relu" }));

model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

const optimizer = tf.train.adam(0.001);
model.compile({
  optimizer,
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

export default model;
