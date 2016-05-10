package com.diffbot.learningfromdata.net;

import java.util.Random;

import com.diffbot.learningfromdata.utils.MathUtils;

public class Neuron {	
	private static Random RANDOM = new Random();
	
	public double[] w;
	
	public Neuron(int numFeatures) {
		w = new double[numFeatures + 1];
		for (int i = 0; i < w.length; i++) {
			w[i] = RANDOM.nextGaussian();
		}
	}
	
	public double dotProduct(double[] x) {
		return MathUtils.dotProduct(w, MathUtils.padBias(x));
	}

}
