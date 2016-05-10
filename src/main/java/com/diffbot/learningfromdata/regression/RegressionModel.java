package com.diffbot.learningfromdata.regression;

public interface RegressionModel {

	public double eval(double[] x);
	public double error(double y, double h);

}
