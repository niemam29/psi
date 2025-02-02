// src/models/modelB.js

const tf = require('@tensorflow/tfjs-node');

/**
 * Tworzy niewielką sieć gęstą (Dense) dla embeddingów wygenerowanych przez MobileNet.
 * Zastosowanie: transfer learning (Model B).
 * @param {number} inputDim - wymiar wektora cech (embeddingu) np. 1024
 * @returns {tf.Sequential} model
 */
function createModelB(inputDim) {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [inputDim],
        units: 64,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 1
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['mae'],
    });

    return model;
}

module.exports = {
    createModelB
};
