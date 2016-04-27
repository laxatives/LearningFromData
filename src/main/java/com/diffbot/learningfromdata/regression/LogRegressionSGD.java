package com.diffbot.learningfromdata.regression;

import java.io.IOException;
import java.util.List;

import com.diffbot.learningfromdata.data.Data;
import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.data.WisconsinBreastCancerData;
import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.Utils;

public class LogRegressionSGD implements RegressionModel {
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
				double[] xPadded = RegressionModel.padBias(x[i]);
				double[] update = MathUtils.scalarProduct(y[i], xPadded);
				double exponent = y[i] * MathUtils.dotProduct(w, xPadded);
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
	
	@Override
	public double eval(double[] x) {
		double max = Math.exp(MathUtils.dotProduct(w, RegressionModel.padBias(x)));
		return max / (1 + max);
	}
	
	@Override
	public double error(double h, double y) {
		double s = Math.log(-h / (h - 1)); // inverse logistic (logit)
		return Math.log(1 + Math.exp(-y * s));
	}
	
	private static final int MAX_ITERATIONS = 1_000;
	private static final double HOLDOUT_PERCENTAGE = 0.2;
	
	public static void main(String[] args) throws IOException {
		LogRegressionSGD model = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, true);		
		List<Labelset> labelSets = Data.split(new WisconsinBreastCancerData().getLabelset(), HOLDOUT_PERCENTAGE);
		Labelset trainSet = labelSets.get(0);
		Labelset testSet = labelSets.get(1);
		
		System.out.println("Training Logistic Regression using Batch Gradient Descent...");
		for (int i = 0; i < MAX_ITERATIONS; i++) {
			System.out.println("Beginning epoch " + i + "...");
			model.train(trainSet.xs, trainSet.ys);
			System.out.println("Train {Var, Bias, MSE}: " + Utils.arrayToString(RegressionModel.getStats(model, testSet.xs, testSet.ys)));	
		}
				
		System.out.println("Test {Var, Bias, MSE}: " + Utils.arrayToString(RegressionModel.getStats(model, testSet.xs, testSet.ys)));
	}
}
