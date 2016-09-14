package com.diffbot.learningfromdata.net;

import java.util.ArrayList;
import java.util.List;

import com.diffbot.learningfromdata.utils.MathUtils;

public class LayerOld {
	private static final double THRESHOLD = 1e-6; 
	
	public final int size;
	private final List<NeuronOld> neurons = new ArrayList<>();
	
	// TODO: allow variable number connections, neuron type (ie RBF)
	public LayerOld(int size, int inputSize) {
		this.size = size;
		for (int i = 0; i < size; i++) {				
			neurons.add(new NeuronOld(inputSize));
		}
	}
	
	public NeuronOld getNeuron(int i) {
		return neurons.get(i);
	}
	
	public double[] forward(double[] x, ActivationFunction f) {
		double[] output = new double[size];
		for (int i = 0; i < output.length; i++) {
			NeuronOld neuron = neurons.get(i);
			double val = f.eval(neuron.dotProduct(x));
			
			output[i] = val;
			neuron.val = val;
		}			
		
		return output;
	}
	
	/**
	 * Returns the outputs of the layer based on the results of the last forward pass.
	 */
	public double[] outputs() {
		double[] output = new double[size];
		for (int i = 0; i < size; i++) {
			NeuronOld neuron = neurons.get(i);
			output[i] = neuron.val;
		}
		return output;
	}
	
	public double[] backward(double[] topLayerGradients, LayerOld topLayer, double[] input, ActivationFunction f) {
		double[] bottomLayerGradients = new double[size];		
		for (int i = 0; i < size; i++) {
			NeuronOld neuron = neurons.get(i);
			double df = f.derivative(neuron.val);
			if (Math.abs(df) < THRESHOLD) {
				continue;
			}
			
			double gradient = 0;
			for (int k = 0; k < topLayer.size; k++) {
				NeuronOld nextNeuron = topLayer.getNeuron(k);				
				gradient += nextNeuron.w[i + 1] * topLayerGradients[k];
			}
			
			if (Double.isNaN(gradient)) {
				System.out.println("Error: Gradient NaN");
				break;
			}
			
			gradient *= df;			
			neuron.updateW(MathUtils.scalarProduct(gradient, input));
							
			bottomLayerGradients[i] = gradient;
		}
		
		return bottomLayerGradients;
	}
}
