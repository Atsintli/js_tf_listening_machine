/*
Desenmarañar código
En qué momento comienza a grabar, cómo esta indicado esto, dónde almacena el audio, dónde los descriptores
Cómo y en que sección extraé las características del audio grabadas
Cómo acceder a los posibles descriptores o a su configuración dentro de la librería
En que lugar se envían las matrices de características a la red neuronal para su entrenamiento
Cómo Intentar implementar una reacción sonora a la clase que detecte
Cómo integrar el análisis de la forma a través de cadenas de Markov
*/

/* If you're feeling fancy you can add interactivity
    to your site with Javascript

// prints "hi" in the browser's dev tools console
console.log('hi');

let recognizer;

function predictWord() {
 // Array of words that the recognizer is trained to recognize.
 const words = recognizer.wordLabels();
 recognizer.listen(({scores}) => {
   // Turn scores into a list of (score,word) pairs.
   scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // Find the most probable word.
   scores.sort((s1, s2) => s2.score - s1.score);
   document.querySelector('#console').textContent = scores[0].word;
 }, {probabilityThreshold: 0.75});
}
*/

// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
let examples = [];

function collect(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label});
   document.querySelector('#console').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}


const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function train() {
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 3);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}

function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 // Add this line.
 buildModel();

  await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
}

app();

async function moveSlider(labelTensor) {
 const label = (await labelTensor.data())[0];
 document.getElementById('console').textContent = label;
 if (label == 2) {
   return;
 }
 let delta = 0.1;
 const prevValue = +document.getElementById('output').value;
 document.getElementById('output').value =
     prevValue + (label === 0 ? -delta : delta);
}

function listen() {
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   document.getElementById('listen').textContent = 'Escucha';
   return;
 }
 toggleButtons(false);
 document.getElementById('listen').textContent = 'Detener';
 document.getElementById('listen').disabled = false;

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
   const probs = model.predict(input);
   const predLabel = probs.argMax(1);
   await moveSlider(predLabel);
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

tf.dispose([input, probs, predLabel]);
