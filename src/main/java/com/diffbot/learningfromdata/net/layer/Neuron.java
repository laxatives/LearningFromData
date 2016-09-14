package com.diffbot.learningfromdata.net.layer;

import java.util.List;

public abstract class Neuron {
    // TODO: create stateless neurons for thread safe parallel execution
    private final int inputSize;
    protected final float learningParam;
    protected List<Float> savedInput;
    protected List<Float> savedGradients;
        
    public Neuron(int inputSize, float learningParam) {
        this.inputSize = inputSize;
        this.learningParam = learningParam;
    }
        
        
    public float forward(List<Float> input) {
        if (input.size() != inputSize) {
            throw new IllegalArgumentException("Expected " + inputSize + " inputs. Got " + input.size());
        }
                
        savedInput = input;
        return doForward(input);
    }
        
    public abstract float doForward(List<Float> input);
        
    public List<Float> backward(List<Float> gradients) {
        savedGradients = doBackward(gradients);
        return savedGradients;
    }
        
    public abstract List<Float> doBackward(List<Float> gradients);
        
    public boolean update() {
        if (savedGradients == null) {
            throw new IllegalStateException("Cannot update neuron without savedGradients computed in backward pass.");
        }
        return doUpdate(savedGradients);            
    }
        
    public abstract boolean doUpdate(List<Float> gradient);
}
