const tf = require('@tensorflow/tfjs-node');
const { createDataset, createDatasetFeatures } = require('./dataUtils'); // Import funkcji do wczytywania danych

/**
 * Ewaluacja modelu na zbiorze testowym
 * @param {string} modelPath - Ścieżka do modelu (np. "my-modelA" lub "my-modelB")
 * @param {string} testCsvPath - Ścieżka do pliku CSV z danymi testowymi
 * @param {string} imagesDir - Katalog z obrazami (np. "data/z")
 */
async function evaluateModel(modelPath, testCsvPath, imagesDir) {
    console.log(`📥 Ładowanie modelu z ${modelPath}...`);
    const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
    console.log("✅ Model załadowany.");

    let dataset;
    if (modelPath.includes("modelA")) {
        console.log("🔹 Model A (CNN) – wczytujemy obrazy...");
        dataset = await createDataset(testCsvPath, imagesDir);
    } else if (modelPath.includes("modelB")) {
        console.log("🔹 Model B (Dense - MobileNet) – generujemy embeddingi...");
        dataset = await createDatasetFeatures(testCsvPath, imagesDir);
    } else {
        throw new Error("Nieznany model! Ścieżka powinna zawierać 'modelA' lub 'modelB'.");
    }

    const { x, y } = dataset;

    // 🚨 Sprawdzamy, czy x istnieje
    if (!x) {
        throw new Error("❌ Błąd: x jest `undefined`. Sprawdź, czy `createDatasetFeatures()` działa poprawnie!");
    }
    console.log(`✅ Dane testowe załadowane. Wymiary x: ${x.shape}, y: ${y.shape}`);

    console.log("📊 Model oczekuje wejścia:", model.input.shape);
    console.log("📊 Wczytane dane testowe mają kształt:", x.shape);

    console.log("🤖 Przewidywanie wartości ratingu...");
    try {
        const predictions = model.predict(x);
        console.log("✅ Predykcja zakończona.");

        const yTrue = y.arraySync();
        const yPred = predictions.arraySync().flat();

        const mse = yTrue.reduce((sum, trueVal, i) => sum + Math.pow(trueVal - yPred[i], 2), 0) / yTrue.length;
        const mae = yTrue.reduce((sum, trueVal, i) => sum + Math.abs(trueVal - yPred[i]), 0) / yTrue.length;

        console.log(`\n🔍 Ewaluacja modelu (${modelPath}):`);
        console.log(`- Mean Squared Error (MSE): ${mse.toFixed(2)}`);
        console.log(`- Mean Absolute Error (MAE): ${mae.toFixed(2)}\n`);

        return { mse, mae };
    } catch (error) {
        console.error("❌ Błąd podczas predykcji:", error);
        console.error("🔹 Możliwe przyczyny: `x` ma zły kształt lub `model.predict(x)` nie obsługuje tego formatu!");
    }
}

module.exports = { evaluateModel };
