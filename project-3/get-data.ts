import tf, { Tensor, Rank } from "@tensorflow/tfjs-node-gpu";
import fs from "fs";
import path from "path";

type DataType = (Tensor<Rank>[] | number[])[];

class Data {
  images: Tensor<Rank>;
  labels: Tensor<Rank>;

  constructor(data: DataType) {
    this.images = tf.concat(data[0]);
    this.labels = tf.oneHot(tf.tensor1d(data[1] as number[], "int32"), 2);
  }
}

const trainImageDir = "./src/data/train";
const testImagesDir = "./src/data/test";

let trainData: DataType, testData: DataType;

const loadImages = (dataDir: string) => {
  const images = [] as Tensor<Rank>[];
  const labels = [] as number[];

  let files = fs.readdirSync(dataDir);
  for (let i = 0; i < files.length; i++) {
    let filePath = path.join(dataDir, files[i]);

    let buffer = fs.readFileSync(filePath);
    let imageTensor = tf.node
      .decodeImage(buffer)
      .resizeNearestNeighbor([28, 28])
      .expandDims();

    images.push(imageTensor);

    const circle = files[i].toLowerCase().endsWith("circle.png");
    const triangle = files[i].toLowerCase().endsWith("triangle.png");

    if (circle === true) {
      labels.push(0);
    } else if (triangle === true) {
      labels.push(1);
    }
  }

  return [images, labels];
};

export const loadData = () => {
  console.log("Loading data...");
  trainData = loadImages(trainImageDir);
  testData = loadImages(testImagesDir);
  console.log("Data loaded");
};

export const getTrainData = () => new Data(trainData);
export const getTestData = () => new Data(testData);
