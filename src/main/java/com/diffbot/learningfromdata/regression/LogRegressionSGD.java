package com.diffbot.learningfromdata.regression;

import java.io.IOException;

import com.diffbot.learningfromdata.data.DataSet.TrainingExamples;
import com.diffbot.learningfromdata.data.WisconsinBreastCancerData;
import com.diffbot.learningfromdata.utils.MathUtils;

public class LogRegressionSGD implements RegressionModel {

	private static final boolean DEBUG = false;
	private static final double NU = 0.1;
	
	public boolean batch = false;
	public double[] w;
	
	/**
	 * Logistic Regression using Stochastic Gradient Descent 
	 * (or Batch Gradient Descent)
	 */
	public LogRegressionSGD(int numFeatures) {
		w = new double[numFeatures];
	}
	
	public LogRegressionSGD(int numFeatures, boolean batch) {
		w = new double[numFeatures];
		this.batch = batch;
	}
	
	/**
	 * Performs one epoch on the given dataset, updating weights as necessary.
	 */
	public void train(double[][] x, double[] y) {
		double[] gradient = new double[w.length];
		if (batch) {
			for (int i = 0; i < x.length; i++) {
				double[] update = MathUtils.scalarProduct(y[i], x[i]);
				double exponent = y[i] * MathUtils.dotProduct(w, x[i]);
				update = MathUtils.scalarProduct(1 / (1 + Math.exp(exponent)), update);
				gradient = MathUtils.sumArrays(gradient, update);
			}
			gradient = MathUtils.scalarProduct(-1 / (double) x.length, gradient);
			updateW(gradient);
		} else {
			// TODO
		}
	}
	
	private void updateW(double[] gradient) {
		gradient = MathUtils.scalarProduct(-NU, gradient);
		w = MathUtils.sumArrays(w, gradient);
	}
	
//	public int train(double[][] x, double[] y, int numIterations) {
//		long start = System.currentTimeMillis();
//		int errors = 0;
//		for (int i = 1; i <= numIterations; i++) {
//			errors = train(x, y); 
//			if (errors == 0) {
//				if (log) {
//					System.out.println(String.format("\tPerceptron converged in %d iterations.", i));
//				}
//				break;
//			}
//			if (log) {
//				System.out.println(String.format("\tCompleted epoch %d with %d mislabeled out of %d total examples.", i, errors, y.length));
//				System.out.println(String.format("\tEstimated Weights: %s\n\t\tBias: %.2f", Utils.arrayToString(weights), bias));
//			}
//		}
//		float took = System.currentTimeMillis() - start;
//		System.out.println(String.format("\tTook %f ms (%.10f per instance)", took, took / y.length));
//		return errors;
//	}
	
//	public double getProb(double[] input) {
//		return MathUtils.dotProduct(input, weights) > bias ? 1 : -1; 
//	}
	
	public double eval(double[] x) {
		return 0;
	}
	
	public static void main(String[] args) throws IOException {
		TrainingExamples es = new WisconsinBreastCancerData().getTrainingExamples();
		RegressionModel model = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, true);
	}
}
