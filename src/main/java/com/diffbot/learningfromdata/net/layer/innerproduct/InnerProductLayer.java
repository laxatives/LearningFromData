package com.diffbot.learningfromdata.net.layer.innerproduct;

import com.diffbot.learningfromdata.net.ActivationFunction;
import com.diffbot.learningfromdata.net.layer.Layer;

public class InnerProductLayer extends Layer {
    public InnerProductLayer(int inputSize, int outputSize, ActivationFunction activationFunction, float learningParam) {
        super(inputSize, outputSize, learningParam, activationFunction);
        for (int i = 0; i < outputSize; i++) {
            // TODO: add a node for bias? currently handled in InnerProductNeuron instead
            // TODO: use 2/(inputSize + outputSize) as init weights
            neurons.add(new InnerProductNeuron(inputSize, activationFunction, learningParam, 1e-2f));
        }
    }
}
