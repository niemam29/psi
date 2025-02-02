// src/models/modelA.js

const tf = require('@tensorflow/tfjs-node');

/**
 * Tworzy prostą sieć CNN do regresji ratingu 0–10 (Model A).
 * @param {number[]} inputShape - np. [224, 224, 3] (wysokość, szerokość, kanały)
 * @returns {tf.Sequential} model
 */
function createModelA(inputShape) {
    const model = tf.sequential();

    // Pierwsza warstwa wejściowa + Convolution
    model.add(tf.layers.conv2d({
        inputShape,     // np. [224, 224, 3]
        kernelSize: 3,
        filters: 16,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Druga warstwa Convolution
    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Spłaszczenie map cech
    model.add(tf.layers.flatten());

    // Warstwa gęsta (ukryta)
    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
    }));

    // Warstwa wyjściowa: 1 neuron dla regresji
    model.add(tf.layers.dense({
        units: 1
    }));

    // Kompilacja modelu
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['mae'],
    });

    return model;
}

module.exports = {
    createModelA
};
