const { evaluateModel } = require('../utils/modelUtils');
const { program } = require('commander');

program
    .option('-m, --model <path>', 'Ścieżka do modelu', 'my-modelA')
    .option('-t, --test-csv <path>', 'Ścieżka do testowego CSV', 'test.csv')
    .option('-i, --images-dir <path>', 'Katalog z obrazami', 'data')
    .action(async (options) => {
        try {
            await evaluateModel(options.model, options.testCsv, options.imagesDir);
        } catch (error) {
            console.error("Błąd podczas ewaluacji:", error);
        }

    })
    .option('--compare', 'Porównanie Modelu A i Modelu B')
    .action(async (options) => {
        if (options.compare) {
            await compareModels();
        }
    });


async function compareModels() {
    console.log("=== Ocena Modelu A ===");
    await evaluateModel('my-modelA', 'test.csv', 'data');

    console.log("\n=== Ocena Modelu B ===");
    await evaluateModel('my-modelB', 'test.csv', 'data');
}

program.parse(process.argv);
