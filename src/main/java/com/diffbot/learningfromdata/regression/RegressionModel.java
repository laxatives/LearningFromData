package com.diffbot.learningfromdata.regression;

import java.util.Arrays;

public interface RegressionModel {

	public double eval(double[] x);
	public double error(double y, double h);
	
	public static double[] padBias(double[] x) {
		double[] p = new double[x.length + 1];
		p[0] = 1;
		System.arraycopy(x, 0, p, 1, x.length);
		return p;
	}
	
	public static double[][] padBias(double[][] x) {
		double[][] p = new double[x.length][x[0].length + 1];
		for (int i = 0; i < x.length; i++) {
			double[] padded = padBias(x[i]);
			p[i] = padded;
		}
		return p;
	}
	
	public static double[] getStats(RegressionModel model, double[][] x, double[] y_t) {             
		double[] estimates = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			estimates[i] = model.eval(x[i]);
		}               
		double avg_estimate = Arrays.stream(estimates).sum() / estimates.length;
		
		double mse = 0;
		double var = 0;
		double bias = 0;
		for (int i = 0; i < x.length; i++) {
			mse += Math.pow(model.error(estimates[i], y_t[i]), 2); 
			var += Math.pow(estimates[i]- avg_estimate, 2);
			bias += avg_estimate - y_t[i];                  
		}               
		
		mse /= x.length;
		var /= x.length;
		bias = Math.pow(bias / x.length, 2);
		               
		return new double[]{var, bias, mse};
	}
	
}
