package com.diffbot.learningfromdata.net;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.diffbot.learningfromdata.data.Data;
import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.Utils;
import com.diffbot.learningfromdata.data.MnistHandwrittenDigitData;

public class NeuralNetwork {	
	private List<Layer> layers = new ArrayList<>();
	private ActivationFunction f;
		
	public NeuralNetwork(int[] shape) {
		this(shape, ActivationFunction.RELU);
	}
	
	/**
	 * First layer size must be equal to number of features.
	 */
	public NeuralNetwork(int[] shape, ActivationFunction f) {
		this.f = f;
		int inputSize = shape[0];
		for (int dim : shape) {
			// TODO: use Radial basis function in final layer
			layers.add(new Layer(dim, inputSize));
			inputSize = dim;
		}
		
	}
	
	private static final int BATCH_SIZE = 256;
	private static final Random RANDOM = new Random();
	
	// this is not thread safe
	public void setup(double[][] xs, double[] ys) {
		for (int i = 0; i < BATCH_SIZE; i++) {
			int ind = RANDOM.nextInt(xs.length);
			double[] x = xs[ind];
			int y = (int) ys[ind];
			double[] guess = forward(x);	
			
			backward(x, guess, y);
		}
	}
	
	public double[] forward(double[] x) {
		double[] input = x;
		for (int i = 0; i < layers.size(); i++) {
			Layer layer = layers.get(i);	
			input = layer.forward(input, f);
		}
		
		return input;
	}
	
	private void backward(double[] x, double[] guess, int y) {		
		Layer outputLayer = layers.get(layers.size() - 1);
		double[] correctLabelWeight = outputLayer.getNeuron(y).w;	

		Layer prevLayer = layers.get(layers.size() - 2);
		double[] input = prevLayer.outputs();
		input = MathUtils.padBias(input);
				
		// compute gradient using last layer output and prev layer inputs
		double[] gradients = new double[outputLayer.size];
		for (int i = 0; i < outputLayer.size; i++) {
			Neuron neuron = outputLayer.getNeuron(i);
			double gradient = grad(input , neuron.w, correctLabelWeight);
			if (Double.isNaN(gradient)) {
				System.out.println("Gradient NaN");
				break;
			}
			neuron.updateW(MathUtils.scalarProduct(f.derivative(neuron.val), MathUtils.scalarProduct(gradient, input)));
			
			gradients[i] = gradient;
		}
		
		for (int i = layers.size() - 2; i >= 0; i--) {						
			Layer topLayer = layers.get(i + 1);			
			Layer bottomLayer = layers.get(i);
			double[] inputs = i > 0 ? layers.get(i - 1).outputs() : x;
			gradients = bottomLayer.backward(gradients, topLayer, inputs, f);
		}
	}
	
	// hinge-loss
	// TODO: try square hinge, cross-entry (soft-max), other loss functions
	public static double loss(double[] output, int y) {
		double correctLabelOutput = output[y];
		double totalLoss = 0;
		for (int i = 0; i < output.length; i++) {
			if (i == y) {
				continue;
			}
			totalLoss += Math.max(0, output[i] - correctLabelOutput + 1);
		}
		return totalLoss;
	}
	
	public static double grad(double[] input, double[] w, double[] correctLabelWeights) {
		if (MathUtils.dotProduct(w, input) - MathUtils.dotProduct(correctLabelWeights, input) > 1)  {
			return 0;
		}
		return 1;
	}

	
	private static final int MAX_ITERATIONS = 1;
	private static final Double HOLDOUT_PERCENTAGE = 0.1;
	
	public static void main(String[] args) {
//		List<Labelset> labelSets = Data.split(new MnistHandwrittenDigitData().getLabelset(), HOLDOUT_PERCENTAGE);
//		Labelset trainSet = labelSets.get(0);
//		Labelset testSet = labelSets.get(1);			
		
//		int[] shape = new int[]{MnistHandwrittenDigitData.NUM_FIELDS, MnistHandwrittenDigitData.NUM_FIELDS, 10};
		int[] shape = new int[]{2, 2};
		
		// logical OR
		double[][] xs = new double[][] {{1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 1}, {0, 0}};
		double[] ys = new double[] {1, 1, 0, 1, 1, 0};
		
		NeuralNetwork nn = new NeuralNetwork(shape);
		
		
		for (int i = 1; i <= MAX_ITERATIONS; i++) {
			nn.setup(xs, ys);			
		}
		
		double loss = 0;
		for (int i = 0; i < xs.length; i++) {
			double[] x = xs[i];
			int y = (int) ys[i];			
			
			double[] guess = nn.forward(x);
			loss += loss(guess, y);
		}
		loss /= xs.length;
		System.out.println("Hinge loss: " + loss);
	}

}
