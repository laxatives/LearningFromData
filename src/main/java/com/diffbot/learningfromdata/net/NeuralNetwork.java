package com.diffbot.learningfromdata.net;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

import com.diffbot.learningfromdata.net.layer.Layer;
import com.diffbot.learningfromdata.net.layer.innerproduct.InnerProductLayer;
import com.google.common.collect.Lists;

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
    
    public void backward(Float err) {
        List<Float> gradients = new ArrayList<>();
        for (int i = 0; i < layers.get(layers.size() - 1).outputSize; i++) {
            gradients.add(1f);
        }
        ListIterator<Layer> li = layers.listIterator(layers.size());
        while (li.hasPrevious()) {
            gradients = li.previous().backward(gradients);
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
    
    public static void main(String[] args) {
        // init network
        NeuralNetwork network = new NeuralNetwork();
        int inputSize = 3;
        int l2Size = 4;
        int outSize = 2;
        float learningParam = 1e-4f;
        
        network.layers.add(new InnerProductLayer(inputSize, l2Size, learningParam));
        network.layers.add(new InnerProductLayer(l2Size, outSize, learningParam));
        
        // init data
        // TODO: create input/label training sample class
        List<List<Float>> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        inputs.add(Lists.newArrayList(0f, 0f, 1f));
        labels.add(0);
        
        inputs.add(Lists.newArrayList(0f, 1f, 1f));
        labels.add(1);
        
        inputs.add(Lists.newArrayList(1f, 0f, 1f));
        labels.add(1);
        
        inputs.add(Lists.newArrayList(1f, 1f, 1f));
        labels.add(0);
        
        // train
        for (int i = 0; i < 50_000; i++) {
            for (int j = 0; j < inputs.size(); j++) {          
                List<Float> input = inputs.get(j);
                int label = labels.get(j);
                
                List<Float> results = network.forward(input);
                float err = svmError(results, label);
                if (Float.isNaN(err)) {
                    throw new IllegalStateException("NaN error at iteration " + i);
                }
                
                System.out.println("svm err: " + err);
                network.backward(err);
                network.update();
            }
        }

        
    } 
    
}
