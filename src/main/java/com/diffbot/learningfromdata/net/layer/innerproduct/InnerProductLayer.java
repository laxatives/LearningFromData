package com.diffbot.learningfromdata.net.layer.innerproduct;

import com.diffbot.learningfromdata.net.layer.Layer;

public class InnerProductLayer extends Layer {
    public InnerProductLayer(int inputSize, int outputSize, float learningParam) {
        super(inputSize, outputSize, learningParam);            
        for (int i = 0; i < outputSize; i++) {
            neurons.add(new InnerProductNeuron(inputSize, learningParam, 2 / (float) (inputSize + outputSize)));
        }
    }
}
