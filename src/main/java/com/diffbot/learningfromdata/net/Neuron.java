package com.diffbot.learningfromdata.net;

import java.util.Random;

import com.diffbot.learningfromdata.utils.MathUtils;

public class Neuron {	
	private static final Double INIT_WEIGHT = 1e-3;
	private static final double ETA = 0.1; // learning parameter
	private static Random RANDOM = new Random();
	
	public double[] w;
	public double val;
	
	/**
	 * Init weights to small numbers drawn from gaussian, bias = 0.
	 */
	public Neuron(int numFeatures) {
		w = new double[numFeatures + 1];
		for (int i = 1; i < w.length; i++) {
			w[i] = INIT_WEIGHT * RANDOM.nextGaussian();
		}
	}
	
	public double dotProduct(double[] x) {
		val = MathUtils.dotProduct(w, MathUtils.padBias(x));
		return val;
	}
	
	public void updateW(double[] gradient) {
		// TODO: support momentum, conjugate descent
		gradient = MathUtils.scalarProduct(-ETA, gradient);
		w = MathUtils.sumArrays(w, gradient);
	}

}