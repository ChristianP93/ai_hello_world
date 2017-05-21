'use strict';

import synaptic from 'synaptic';
import mnist from 'mnist';

const set = mnist.set(700, 20);

const trainingSet = set.training;
const testSet = set.test;


const Layer = synaptic.Layer;
const Network = synaptic.Network;
const Trainer = synaptic.Trainer;

//TODO problematiche di bilanciamento

const inputLayer = new Layer(784);
const hiddenLayer = new Layer(100); // TODO perchè 100?
const outputLayer = new Layer(10); //TODO o 3 perchè in binario per fare 10 servono 3 bit

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer], //TODO array? Spiegazione quali sono i pro di aggiungere piu livelli nascosti?
    output: outputLayer
});

const trainer = new Trainer(myNetwork);
trainer.train(trainingSet, {
    rate: .2,
    iterations: 30,
    error: .1,
    shuffle: true,
    log: 1,
    cost: Trainer.cost.CROSS_ENTROPY
});
console.log(myNetwork.activate(testSet[0].input));
console.log(testSet[0].output);
