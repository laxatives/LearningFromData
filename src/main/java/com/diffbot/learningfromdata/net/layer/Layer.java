package com.diffbot.learningfromdata.net.layer;

import com.diffbot.learningfromdata.net.ActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public abstract class Layer {
    public final int inputSize;
    public final int outputSize;
    protected final float learningParam;
    protected final ActivationFunction activationFunction;
    
    protected List<Neuron> neurons = new ArrayList<>();
    
    public Layer(int inputSize, int outputSize, float learningParam, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningParam = learningParam;
        this.activationFunction = activationFunction;
    }
    
    public List<Float> forward(List<Float> inputs) {
        if (inputs.size() != inputSize) {
            throw new IllegalArgumentException("Expected " + inputSize + " inputs. Got " + inputs.size());
        }

        List<Float> results = neurons.stream()
                .map(n -> n.forward(inputs))
                .collect(Collectors.toList());

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
            if (localGradients.size() != inputSize) {
                throw new IllegalArgumentException("Expected " + inputSize + " local gradients. Got " + localGradients.size());
            }
            
            // add to the gradient of each input for each neuron in this layer
            for (int i = 0; i < localGradients.size(); i++) {
                gradients.set(i, gradients.get(i) + activationFunction.derivative(localGradients.get(i)));
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
    
    public String debug() {
        StringBuilder debug = new StringBuilder();
        debug.append("\tLayer:\n");
        for (Neuron n : neurons) {
            debug.append(n.debug());
        }
        return debug.toString();
    }
}
