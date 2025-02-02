// src/cli/predict.js

const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const { loadImageTensor } = require('../utils/dataUtils');

/**
 * Funkcja wywoływana przez komendę "predict" w CLI.
 *
 * @param {Object} options - parametry przekazane z commander (modelType, image, itd.)
 */
async function predict(options) {
    const {
        modelType,
        image,
        channels = 3
    } = options;

    console.log(`\n--- Uruchamiam predykcję. Model = ${modelType} ---`);
    console.log(`Obraz: ${image}\n`);

    if (modelType === 'A') {
        const model = await tf.loadLayersModel('file://./my-modelA/model.json');

        const imgTensor = await loadImageTensor(image, 224, channels);

        const prediction = model.predict(imgTensor);
        const ratingVal = (await prediction.data())[0];

        console.log(`Przewidywany rating (Model A) = ${ratingVal.toFixed(2)}`);

        imgTensor.dispose();
        prediction.dispose();

    } else if (modelType === 'B') {
        const model = await tf.loadLayersModel('file://./my-modelB/model.json');

        const mobileNetModel = await mobilenet.load({
            version: 2,
            alpha: 1.0
        });

        const imgTensor = await loadImageTensor(image, 224, channels);
        const embedding = mobileNetModel.infer(imgTensor, 'conv_preds');

        const prediction = model.predict(embedding);
        const ratingVal = (await prediction.data())[0];

        console.log(`Przewidywany rating (Model B) = ${ratingVal.toFixed(2)}`);

        imgTensor.dispose();
        embedding.dispose();
        prediction.dispose();

    } else {
        console.error('Nieznany typ modelu! Użyj "A" lub "B".');
    }
}

module.exports = {
    predict
};
