// src/cli/train.js

const tf = require('@tensorflow/tfjs-node');
const {
    createDataset,
    createDatasetFeatures
} = require('../utils/dataUtils');
const { createModelA } = require('../models/modelA');
const { createModelB } = require('../models/modelB');

/**
 * Funkcja wywoływana przez komendę "train" w CLI.
 *
 * @param {Object} options - parametry przekazane z commander (modelType, epochs, batchSize, etc.)
 */
async function train(options) {
    const {
        modelType,
        epochs,
        batchSize,
        csvPath,
        imagesDir,
        channels = 3  // domyślnie 3 (RGB)
    } = options;

    console.log(`\n--- Rozpoczynam trenowanie. Model = ${modelType} ---`);
    console.log(`CSV:       ${csvPath}`);
    console.log(`Images:    ${imagesDir}`);
    console.log(`Epochs:    ${epochs}`);
    console.log(`BatchSize: ${batchSize}\n`);

    if (modelType === 'A') {
        //
        // Model A: CNN od zera
        //
        console.log('> Wczytywanie datasetu dla Modelu A...');
        const { x, y } = await createDataset(csvPath, imagesDir, 224, channels);

        console.log('> Tworzenie Modelu A...');
        // inputShape => [224, 224, channels]
        const model = createModelA([224, 224, channels]);

        console.log('> Trenowanie Modelu A...');
        await model.fit(x, y, {
            epochs,
            batchSize,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1} / ${epochs} | loss=${logs.loss.toFixed(4)} | mae=${logs.mae.toFixed(4)}`);
                }
            }
        });

        console.log('> Zapisywanie Modelu A do my-modelA...');
        await model.save('file://./my-modelA');

        x.dispose();
        y.dispose();
        console.log('Gotowe. Model A zapisany.\n');

    } else if (modelType === 'B') {
        console.log('> Generowanie embeddingów (MobileNet) dla Modelu B...');
        const { x, y } = await createDatasetFeatures(csvPath, imagesDir, 224, channels);

        console.log('> Tworzenie Modelu B...');
        const inputDim = 1280;
        const model = createModelB(inputDim);

        console.log('> Trenowanie Modelu B...');
        await model.fit(x, y, {
            epochs,
            batchSize,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1} / ${epochs} | loss=${logs.loss.toFixed(4)} | mae=${logs.mae.toFixed(4)}`);
                }
            }
        });

        console.log('> Zapisywanie Modelu B do my-modelB...');
        await model.save('file://./my-modelB');

        x.dispose();
        y.dispose();
        console.log('Gotowe. Model B zapisany.\n');

    } else {
        console.error('Nieznany typ modelu! Użyj "A" lub "B".');
    }
}

module.exports = {
    train
};
