// src/utils/dataUtils.js

const fs = require('fs');
const csv = require('csv-parser');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const path = require('path');

/**
 * 1) Wczytuje plik CSV i zwraca tablicę obiektów
 *    w formacie: [{ filename: 'image1.jpg', rating: 8 }, ... ]
 */
async function loadCsvData(csvPath) {
    return new Promise((resolve, reject) => {
        const results = [];
        fs.createReadStream(csvPath)
            .pipe(csv({ separator: ';' }))
            .on('data', (data) => {
                // data to np. { filename: 'image1.jpg', rating: '8' }
                // Upewniamy się, że rating jest liczbą
                const row = {
                    filename: data.filename,
                    rating: parseFloat(data.rating)
                };
                results.push(row);
            })
            .on('end', () => resolve(results))
            .on('error', (err) => reject(err));
    });
}

/**
 * 2) Wczytuje pojedynczy obraz z dysku (imagePath),
 *    zmienia rozmiar do targetSize x targetSize,
 *    konwertuje do tensora 4D [1, height, width, channels].
 *
 *    channels = 3 (RGB) – standardowo do MobileNet / do klasycznych CNN
 *    Możesz zmienić na 4, jeśli chcesz RGBA, ale zwykle wystarczą 3 kanały.
 */
async function loadImageTensor(imagePath, targetSize = 224, channels = 3) {
    let pipeline = sharp(imagePath)
        .resize(targetSize, targetSize)
        .toFormat('png')
        .raw();

    pipeline = pipeline.removeAlpha()

    const imgBuffer = await pipeline.toBuffer();

    const imgTensor = tf.tensor4d(
        imgBuffer,
        [1, targetSize, targetSize, channels],
        'int32'
    );

    // Normalizujemy do [0,1], dzieląc przez 255
    const normalized = imgTensor.div(tf.scalar(255));
    return normalized;
}

/**
 * 3) createDataset - dla Modelu A (trenowanego od zera).
 *    Wczytuje cały CSV (z pliku csvPath), a następnie dla każdego rekordu:
 *      - ładuje obraz (imagesDir + filename),
 *      - wrzuca do tablicy imageTensors
 *      - rating do tablicy ratings
 *    Na koniec scala to w dwa duże tensory: x (obrazy), y (ratingi).
 */
async function createDataset(csvPath, imagesDir, targetSize = 224, channels = 3) {
    const records = await loadCsvData(csvPath);

    if (records.length === 0) {
        throw new Error("❌ Plik CSV nie zawiera żadnych danych!");
    }

    console.log(`📂 Wczytano ${records.length} rekordów z CSV.`);

    const imageTensors = [];
    const ratings = [];

    for (const { filename, rating } of records) {
        const cleanedFilename = filename.replace(/^\.\//, '');
        const imagePath = path.join(imagesDir, cleanedFilename);

        try {
            const imgTensor = await loadImageTensor(imagePath, targetSize, channels);
            imageTensors.push(imgTensor);
            ratings.push(rating);
        } catch (error) {
            console.error(`❌ Błąd wczytywania obrazu: ${imagePath}`, error);
        }
    }

    if (imageTensors.length === 0) {
        throw new Error("❌ Brak poprawnych obrazów! Sprawdź ścieżki i format CSV.");
    }

    const x = tf.concat(imageTensors);
    const y = tf.tensor1d(ratings);

    console.log(`✅ Stworzone tensory: x=${x.shape}, y=${y.shape}`);

    imageTensors.forEach(t => t.dispose());
    return { x, y };
}




/**
 * 4) createDatasetFeatures - dla Modelu B (transfer learning).
 *    - Wczytuje CSV, ładuje model MobileNet, a następnie:
 *      * dla każdego obrazu tworzy tensor
 *      * przepuszcza przez MobileNet, wyciągając tzw. "embedding" (cechy)
 *      * zbiera embeddingi do tablicy (features) i ratingi do (ratings)
 *    - Na koniec zwraca X (tensor2d) i y (tensor1d).
 */
async function createDatasetFeatures(csvPath, imagesDir, targetSize = 224, channels = 3) {
    const records = await loadCsvData(csvPath);

    if (records.length === 0) {
        throw new Error("❌ Plik CSV nie zawiera żadnych danych!");
    }

    console.log(`📂 Wczytano ${records.length} rekordów z CSV.`);

    const mobileNetModel = await mobilenet.load({
        version: 2,
        alpha: 1.0
    });

    const features = [];
    const ratings = [];

    for (const { filename, rating } of records) {
        const cleanedFilename = filename.replace(/^\.\//, '');
        const imagePath = path.join(imagesDir, cleanedFilename);

        try {
            const imgTensor = await loadImageTensor(imagePath, targetSize, channels);
            const embedding = mobileNetModel.infer(imgTensor, 'conv_preds');

            // **Sprawdzenie embeddingu przed dodaniem**
            const embeddingData = (await embedding.array())[0];

            if (!embeddingData || embeddingData.length !== 1280 || embeddingData.some(isNaN)) {
                console.warn(`⚠️ Pominięto ${filename}: embedding jest pusty lub błędny.`);
                continue; // ❌ Pomiń ten rekord
            }

            console.log(`✅ Embedding dla ${filename}: ${embeddingData.slice(0, 5)}... (${embeddingData.length} wartości)`);

            features.push(embeddingData);
            ratings.push(rating);

            imgTensor.dispose();
            embedding.dispose();
        } catch (error) {
            console.error(`❌ Błąd wczytywania obrazu: ${imagePath}`, error);
        }
    }

    if (features.length === 0) {
        throw new Error("❌ Brak poprawnych embeddingów! Sprawdź, czy obrazy się wczytują.");
    }

    console.log(`✅ Stworzone tensory: X - ${features.length} próbek, y - ${ratings.length} wartości`);

    // **Ostateczne sprawdzenie formatu**
    if (!Array.isArray(features) || !Array.isArray(features[0])) {
        console.error("❌ Błąd: `features` nie jest poprawną tablicą!");
        console.error(features);
        throw new Error("❌ `features` musi być tablicą tablic liczb!");
    }

    if (features.length === 0 || ratings.length === 0) {
        throw new Error("❌ Brak poprawnych embeddingów! Sprawdź, czy obrazy się wczytują.");
    }

    const validFeatures = features.filter(f => f.length === 1280);
    const validRatings = ratings.slice(0, validFeatures.length);

    if (validFeatures.length === 0) {
        throw new Error("❌ Wszystkie embeddingi mają błędne rozmiary! Sprawdź MobileNet.");
    }

    const x = tf.tensor2d(validFeatures, [validFeatures.length, 1280]);
    const y = tf.tensor1d(validRatings);

    return { x, y };
}

module.exports = {
    loadCsvData,
    loadImageTensor,
    createDataset,
    createDatasetFeatures
};
