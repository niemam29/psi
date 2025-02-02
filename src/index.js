#!/usr/bin/env node
const { program } = require('commander');
const { train } = require('./cli/train');
const { predict } = require('./cli/predict');

program
    .command('train')
    .description('Trenuj model')
    .option('-m, --model-type <type>', 'Wybór modelu: A lub B', 'A')
    .option('-e, --epochs <number>', 'Liczba epok', '10')
    .option('-b, --batch-size <number>', 'Batch size', '8')
    .option('--csv-path <path>', 'Ścieżka do pliku CSV', 'data/labels.csv')
    .option('--images-dir <path>', 'Ścieżka do folderu z obrazami', 'data')
    .action(async (cmd) => {
        const epochs = parseInt(cmd.epochs, 10);
        const batchSize = parseInt(cmd.batchSize, 10);

        await train({
            modelType: cmd.modelType,
            epochs,
            batchSize,
            csvPath: cmd.csvPath,
            imagesDir: cmd.imagesDir
        });
    });

program
    .command('predict')
    .description('Przewiduj rating dla pojedynczego obrazu')
    .requiredOption('-m, --model-type <type>', 'Wybór modelu: A lub B')
    .requiredOption('-i, --image <path>', 'Ścieżka do obrazu')
    .action(async (cmd) => {
        await predict({
            modelType: cmd.modelType,
            image: cmd.image
        });
    });

program.parse(process.argv);
