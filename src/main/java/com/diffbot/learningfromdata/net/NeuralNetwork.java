package com.diffbot.learningfromdata.net;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.diffbot.learningfromdata.data.Data;
import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.data.MnistHandwrittenDigitData;

public class NeuralNetwork {	
	private List<List<Neuron>> neurons = new ArrayList<>();
	private ActivationFunction f;
	
	public enum ActivationFunction {
		TANH {
			@Override
			public double eval(double x) {
				return Math.tanh(x);
			}
		},
		RELU {
			@Override
			public double eval(double x) {
				return Math.max(0, x);
			}
		},
		LEAKY_RELU {
			private static final double a = 0.01;
			@Override
			public double eval(double x) {
				return x < 0 ? a * x : x; 
			}
		};
		public abstract double eval(double x);
	}
	
	public NeuralNetwork(int[] shape) {
		this(shape, ActivationFunction.RELU);
	}
	
	public NeuralNetwork(int[] shape, ActivationFunction f) {
		this.f = f;
		int inputSize = 1;
		for (int dim : shape) {
			List<Neuron> layer = new ArrayList<>();
			
			for (int neuronIdx = 0; neuronIdx < dim; neuronIdx++) {				
				layer.add(new Neuron(inputSize));
			}

			neurons.add(layer);
			inputSize = dim;
		}
	}
	
	public int forward(double[] x) {
		double[] input = x;
		for (int i = 0; i < neurons.size(); i++) {
			List<Neuron> layer = neurons.get(i);			
			double[] output = new double[layer.size()];
			
			for (int j = 0; j < output.length; j++) {
				output[j] = f.eval(layer.get(j).dotProduct(input));
			}
			
			input = output;
		}
		
		int result = 0;
		double maxWeight = Double.MIN_VALUE;
		for (int i = 0; i < input.length; i++) {
			if (input[i] > maxWeight) {
				result = i;
				maxWeight = input[i];
			}
		}
		
		return result;
	}
	
	public static int backward() {
		// TODO
		return 0;
	}
	
	
	private static final Double HOLDOUT_PERCENTAGE = 0.1;
	
	public static void main(String[] args) throws IOException {
		List<Labelset> labelSets = Data.split(new MnistHandwrittenDigitData().getLabelset(), HOLDOUT_PERCENTAGE);
		Labelset trainSet = labelSets.get(0);
		Labelset testSet = labelSets.get(1);
		
		// 2 hidden layers of input size
		int[] shape = new int[]{MnistHandwrittenDigitData.NUM_FIELDS, MnistHandwrittenDigitData.NUM_FIELDS, MnistHandwrittenDigitData.NUM_FIELDS, 10};
		NeuralNetwork nn = new NeuralNetwork(shape);
		for (int i = 0; i < trainSet.xs.length; i++) {
			double[] x = trainSet.xs[i];
			double y = trainSet.ys[i];
			nn.forward(x);
		}
	}

}
