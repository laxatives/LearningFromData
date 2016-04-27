package com.diffbot.learningfromdata.regression;

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
	
}
