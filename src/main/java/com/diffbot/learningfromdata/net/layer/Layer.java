package com.diffbot.learningfromdata.net.layer;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {
    public final int inputSize;
    public final int outputSize;
    protected final float learningParam;
    
    protected List<Neuron> neurons = new ArrayList<>();
    
    public Layer(int inputSize, int outputSize, float learningParam) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningParam = learningParam;
    }
    
    public List<Float> forward(List<Float> inputs) {
        if (inputs.size() != inputSize) {
            throw new IllegalArgumentException("Expected " + inputSize + " inputs. Got " + inputs.size());
        }

        List<Float> results = new ArrayList<>();
        for (Neuron n : neurons) {
            results.add(n.forward(inputs));
        }

        return results;
    }
    
    public List<Float> backward(List<Float> nextLayerGradients) {        
        if (nextLayerGradients.size() != outputSize) {
            throw new IllegalArgumentException("Expected " + outputSize + " gradients. Got " + nextLayerGradients.size());
        }
        
        List<Float> gradients = new ArrayList<>();
        for (int i = 0; i < inputSize; i++) {
            gradients.add(0f);
        }
        
        for (Neuron n : neurons) {
            List<Float> localGradients = n.backward(nextLayerGradients);
            for (int i = 0; i < localGradients.size(); i++) {
                gradients.set(i, gradients.get(i) + localGradients.get(i));
            }
        }
        
        return gradients;
    }
    
    public boolean update() {
        for (Neuron n : neurons) {
            if (!n.update()) {
                // something bad happened
                return false;
            }
        }
        
        return true;
    }
}
