package com.diffbot.learningfromdata.net;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.diffbot.learningfromdata.data.Data;
import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.utils.Utils;
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
			@Override
			public double derivative(double x) {
				return 1 - Math.pow(Math.tanh(x), 2);
			}
		},
		RELU {
			@Override
			public double eval(double x) {
				return Math.max(0, x);
			}

			@Override
			public double derivative(double x) {
				return x < 0 ? 0 : 1;
			}
		},
		LEAKY_RELU {
			private static final double a = 0.01;
			@Override
			public double eval(double x) {
				return x < 0 ? a * x : x; 
			}
			@Override
			public double derivative(double x) {
				return x < 0 ? a : 1;
			}
		};
		public abstract double eval(double x);
		public abstract double derivative(double x);
	}
	
	public NeuralNetwork(int[] shape) {
		this(shape, ActivationFunction.RELU);
	}
	
	/**
	 * First layer size must be equal to number of features.
	 */
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
	
	public double[] eval(double[] x) {
		double[] input = x;
		for (int i = 0; i < neurons.size(); i++) {
			List<Neuron> layer = neurons.get(i);			
			double[] output = new double[layer.size()];
			
			for (int j = 0; j < output.length; j++) {
				output[j] = f.eval(layer.get(j).dotProduct(input));
			}
			
			input = output;
		}
		
		return input;
	}
	
	public void train(double[][] xs, double[] ys) {
		for (int i = 0; i < xs.length; i++) {
			// TODO: use minibatch
			double[] x = xs[i];
			double y = ys[i];
			double[] guess = eval(x);	
			
			backprop(guess, y);
		}
	}
	
	private void backprop(double[] guess, double y) {
		double[] output = guess;
		for (int i = neurons.size() - 1; i >= 0; i--) {
			List<Neuron> layer = neurons.get(i);

			double[] input = new double[layer.size()];
			
			for (int j = 0; j < input.length; j++) {
				Neuron neuron = layer.get(j);
				double[] gradient = null; // need edge values from forward pass
				input[j] = f.derivative(layer.get(j).dotProduct(output));
			}
			
			output = input;
		}
	}
	
	// square hinge loss
	public static double loss(double[] output, double y) {
		double loss = 0;
		int correctIndex = (int) y;
		for (int i = 0; i < output.length; i++) {
			if (i == correctIndex) {
				continue;
			}
			loss += Math.pow(Math.max(0, 1 - output[i]), 2);
		}
		return loss;
	}
	
	private static final int MAX_ITERATIONS = 200;
	private static final Double HOLDOUT_PERCENTAGE = 0.1;
	
	public static void main(String[] args) throws IOException {
		List<Labelset> labelSets = Data.split(new MnistHandwrittenDigitData().getLabelset(), HOLDOUT_PERCENTAGE);
		Labelset trainSet = labelSets.get(0);
		Labelset testSet = labelSets.get(1);
		
		// 2 hidden layers of input size
		int[] shape = new int[]{MnistHandwrittenDigitData.NUM_FIELDS, MnistHandwrittenDigitData.NUM_FIELDS, MnistHandwrittenDigitData.NUM_FIELDS, 10};
		NeuralNetwork nn = new NeuralNetwork(shape);
		for (int i = 1; i <= MAX_ITERATIONS; i++) {
			nn.train(trainSet.xs, trainSet.ys);			
		}
		
		double loss = 0;
		for (int i = 0; i < testSet.xs.length; i++) {
			double[] x = testSet.xs[i];
			double y = testSet.ys[i];
			
			double[] guess = nn.eval(x);
			System.out.println("" + loss(guess, y) + " = " +y + ": " +  Utils.arrayToString(x));
		}
		System.out.println("Average loss: " + loss / (float) testSet.xs.length);
	}

}
