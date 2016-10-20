package com.diffbot.learningfromdata.net;

import com.diffbot.learningfromdata.net.layer.Layer;
import com.diffbot.learningfromdata.net.layer.innerproduct.InnerProductLayer;
import com.google.common.collect.Lists;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

public class NeuralNetwork {
    List<Layer> layers = new ArrayList<>();    
    
    public List<Float> forward(List<Float> inputs) {
        List<Float> results = null;
        for (Layer layer : layers) {
            results = layer.forward(inputs);
            inputs = results;
        }

        return results;               
    }
    
    // TODO: accept different loss functions/regularization
    public void backward(List<Float> results, int correctIndex) {
        List<Float> gradients = new ArrayList<>();
        for (int i = 0; i < results.size(); i++) {
            gradients.add(0f);
        }
        
        for (int i = 0; i < results.size(); i++) {
            if (i == correctIndex) {
                continue;
            } else if (results.get(i) - results.get(correctIndex) + 1 > 0) {
                gradients.set(i, 1f);
                gradients.set(correctIndex, gradients.get(correctIndex) - 1f);
            }
        }
        ListIterator<Layer> li = layers.listIterator(layers.size());
        while (li.hasPrevious()) {
            Layer layer = li.previous();
            gradients = layer.backward(gradients);
        }
    }       
        
    public boolean update() {
        for (Layer layer : layers) {
            if (!layer.update()) {
                // something bad happened
                return false;
            }
        }
        return true;
    }
    
    public String debug() {
        StringBuilder sb = new StringBuilder();
        sb.append("Network:\n");
        for (Layer layer : layers) {
            sb.append(layer.debug());
        }
        return sb.toString();
    }
    
    // TODO: move to error function package
    public static float svmError(List<Float> results, int correctIndex) {
        if (correctIndex >= results.size()) {
            throw new IllegalArgumentException("Cannot have correctIndex " + correctIndex + " for results " + results);
        }
        float err = 0;
        for (int i = 0; i < results.size(); i++) {
            if (i == correctIndex) {
                continue;
            }
            err += Math.max(0, results.get(i) - results.get(correctIndex) + 1);
        }
        
        return err;
    }
    
    public static float softMaxError(List<Float> results, int correctIndex) {
        if (correctIndex >= results.size()) {
            throw new IllegalArgumentException("Cannot have correctIndex " + correctIndex + " for results " + results);
        }
        float err = 0;
        for (int i = 0; i < results.size(); i++) {
            if (i == correctIndex) {
                continue;
            }
            err += Math.max(0, results.get(i) - results.get(correctIndex) + 1);
        }
        
        return err;
    }
    
    public static void main(String[] args) {        
        // init network
        NeuralNetwork network = new NeuralNetwork();
        int inputSize = 2;
        int outputSize = 2;
        float learningParam = 1e-3f;
        
        network.layers.add(new InnerProductLayer(inputSize, outputSize, ActivationFunction.RELU, learningParam));
        System.out.println(network.debug());
        
        // init data
        // TODO: create input/label training sample class
        List<List<Float>> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        inputs.add(Lists.newArrayList(-1f, 1f));
        labels.add(1);
//        
//        inputs.add(Lists.newArrayList(0f, 1f));
//        labels.add(1);
//
        inputs.add(Lists.newArrayList(1f, -1f));
        labels.add(0);
//        
//        inputs.add(Lists.newArrayList(1f, 0f));
//        labels.add(0);
        
        // train
        for (int i = 0; i < 200_000; i++) {
            for (int j = 0; j < inputs.size(); j++) {          
                List<Float> input = inputs.get(j);
                int label = labels.get(j);
                
                List<Float> results = network.forward(input);
                float err = svmError(results, label);
                if (Float.isNaN(err)) {
                    throw new IllegalStateException("NaN error at iteration " + i);
                }
                
                if (i % 10_000 == 0) System.out.println(i + "\tsvm err @" + j +": " + err);
                network.backward(results, label);
                if (!network.update()) {
                    throw new RuntimeException("Unexpected error in network update.");
                }
            }
        }
        
        System.out.println(network.debug());
        
        // test
        System.out.println("TESTING");
        for (int j = 0; j < inputs.size(); j++) {          
            List<Float> input = inputs.get(j);
            int label = labels.get(j);
            
            List<Float> results = network.forward(input);
            float err = svmError(results, label);
            System.out.println(input + " => " + results + ", true label: " + label);
            System.out.println("\tsvm err: " + err);
        }         
    } 
    
}
