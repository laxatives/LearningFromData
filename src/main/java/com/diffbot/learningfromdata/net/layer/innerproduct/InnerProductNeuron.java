package com.diffbot.learningfromdata.net.layer.innerproduct;

import com.diffbot.learningfromdata.net.layer.Neuron;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class InnerProductNeuron extends Neuron {
    private List<Float> weights = new ArrayList<>();
    private float bias;
    
    public InnerProductNeuron(int inputSize, float learningParam, float initWeightRange) {
        super(inputSize, learningParam);   
        for (int i = 0; i < inputSize; i++) {
            weights.add(initWeight(initWeightRange));
        }
        bias = initWeight(initWeightRange);
    }
                    
    private static float initWeight(float initWeightRange) {
        // TODO: use seed for predictable debugging
        Random r = new Random();
        return initWeightRange * (r.nextFloat() - 0.5f);
    }

    @Override
    public float doForward(List<Float> input) {
        float result = 0;
        for (int i = 0; i < weights.size(); i++) {
            result += input.get(i) * weights.get(i);
        }
        result -= bias;
        return result;
    }
    
    /**
     * Don't pass the last gradient (used for bias) to the previous layer.
     */
    @Override
    public List<Float> backward(List<Float> nextLayerGradients) {
        savedGradients = doBackward(nextLayerGradients);
        return savedGradients.subList(0, savedGradients.size() - 1);           
    }

    @Override
    public List<Float> doBackward(List<Float> nextLayerGradients) {
        // init gradients to 0 (one for each weight parameter + one for bias)
        List<Float> gradients = new ArrayList<>();                
        for (int i = 0; i < weights.size() + 1; i++) {
            gradients.add(0f);
        }
        
        for (int i = 0; i < weights.size() + 1; i++) {
            if (i < savedInput.size()) {
                // contribute to the weight gradient
                for (float nextLayerGradient : nextLayerGradients) {
                    gradients.set(i, gradients.get(i) + nextLayerGradient * savedInput.get(i));
                }
            } else {
                // contribute to the bias gradient (always has input value 1)
                for (float nextLayerGradient : nextLayerGradients) {
                    gradients.set(i, gradients.get(i) + nextLayerGradient);
                }
            }
        }
            
        return gradients;                                
    } 

    @Override
    public boolean doUpdate(List<Float> gradients) {
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningParam * gradients.get(i));
        }
            
        bias -= learningParam * gradients.get(weights.size());
            
        return true;
    }
    
    @Override
    public String debug() {
        return "\t\t" + weights.toString() + "; " + bias + "\n";
    }
}
