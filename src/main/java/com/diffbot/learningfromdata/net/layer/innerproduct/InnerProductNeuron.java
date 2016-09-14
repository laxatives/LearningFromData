package com.diffbot.learningfromdata.net.layer.innerproduct;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.diffbot.learningfromdata.net.layer.Neuron;

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
        result += bias;
        return result;
    }
    
    /**
     * Don't pass the last gradient (used for bias) to the next layer.
     */
    @Override
    public List<Float> backward(List<Float> gradients) {
        savedGradients = doBackward(gradients);
        List<Float> passedGradients = savedGradients.subList(0, savedGradients.size() - 1);
            
        gradientCheck(passedGradients);
            
        return passedGradients;
    }

    @Override
    public List<Float> doBackward(List<Float> errors) {
        // init gradients to 0 (one for each weight parameter + one for bias)
        List<Float> gradients = new ArrayList<>();                
        for (int i = 0; i < weights.size() + 1; i++) {
            gradients.add(0f);
        }
            
        // for each error (one for each neuron in the following layer)
        for (float err : errors) {
            // contribute to the weight gradient
            for (int i = 0; i < savedInput.size(); i++) {
                gradients.set(i, gradients.get(i) + err * savedInput.get(i));
            }
            // contribute to the bias gradient
            gradients.set(weights.size(), gradients.get(weights.size()) + err);
        }                                
            
        return gradients;                                
    }
        
    private static final float DELTA = 1e-5f;
    private static final float THRESHOLD = 1e-3f;
        
    private void gradientCheck(List<Float> gradients) {                
        for (int i = 0; i < gradients.size(); i++) {
            List<Float> shiftedLeft = new ArrayList<>();
            List<Float> shiftedRight = new ArrayList<>();
            for (int j = 0; j < savedInput.size(); j++) {
                if (i == j) {
                    shiftedLeft.add(savedInput.get(j) - DELTA);
                    shiftedRight.add(savedInput.get(j) + DELTA);
                } else {
                    shiftedLeft.add(savedInput.get(j));
                    shiftedRight.add(savedInput.get(j));
                }
            }
            float resultLeft = doForward(shiftedLeft);
            float resultRight = doForward(shiftedRight);
            float numericGradient = (resultRight - resultLeft) / (2 * DELTA);                    
            if (Math.abs(gradients.get(i) - numericGradient) > THRESHOLD) {
                System.out.println(gradients + ", " + i);
                throw new IllegalStateException("Failed gradient check: gradient=" + gradients.get(i) + " != numericGradient=" + numericGradient);
            }
        }
    }

    @Override
    public boolean doUpdate(List<Float> gradients) {
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningParam * gradients.get(i));
        }
            
        bias -= learningParam * gradients.get(weights.size());
            
        return true;
    }
}
