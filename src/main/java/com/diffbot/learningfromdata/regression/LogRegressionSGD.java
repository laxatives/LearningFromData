package com.diffbot.learningfromdata.regression;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.diffbot.learningfromdata.data.Data;
import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.data.WisconsinBreastCancerData;
import com.diffbot.learningfromdata.utils.MathUtils;

public class LogRegressionSGD implements RegressionModel {
	private static final double ETA = 0.1;
	
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
			for (int i = 0; i < x.length; i++) {
				double[] xPadded = RegressionModel.padBias(x[i]);
				double[] update = MathUtils.scalarProduct(y[i], xPadded);
				double exponent = y[i] * MathUtils.dotProduct(w, xPadded);
				update = MathUtils.scalarProduct(1 / (1 + Math.exp(exponent)), update);
				gradient = MathUtils.sumArrays(gradient, update);
				gradient = MathUtils.scalarProduct(-1 / (double) x.length, gradient);
				updateW(gradient);
			}
		}
	}
	
	// TODO: this should depend on error
	private void updateW(double[] gradient) {
		gradient = MathUtils.scalarProduct(-ETA, gradient);
		w = MathUtils.sumArrays(w, gradient);
	}
	
	@Override
	public double eval(double[] x) {
		double max = Math.exp(MathUtils.dotProduct(w, RegressionModel.padBias(x)));
		return max / (1 + max);
	}
	
	@Override
	public double error(double h, double y) {
		double s = Math.log(-h / (h - 1)); // logit function (inverse logistic)
		return Math.log(1 + Math.exp(-y * s));
	}
	
	public static Map<String, Double> getStats(LogRegressionSGD model, double[][] x, double[] y_t) {             
		double err = 0;
		Map<Double, List<Integer>> counts = new TreeMap<>();
		for (Double threshold : Arrays.asList(0.01, 0.25, 0.50, 0.75, 0.99)) {
			counts.put(threshold, new ArrayList<>(Collections.nCopies(4, 0))); // [tp, fp, tn, fn]
		}
		for (int i = 0; i < x.length; i++) {
			double truth = y_t[i];
			double estimate = model.eval(x[i]);			
			err += model.error(estimate, truth);
			for (Double threshold : counts.keySet()) {
				List<Integer> thresholdCounts = counts.get(threshold);			
				if (estimate > threshold) { // positive
					if (truth > 0.99999) { // true
						thresholdCounts.set(0, thresholdCounts.get(0) + 1);
					} else { // false
						thresholdCounts.set(1, thresholdCounts.get(1) + 1);
					}
				} else { // negative
					if (truth > 0.99999) { // false
						thresholdCounts.set(3, thresholdCounts.get(3) + 1);
					} else { // true
						thresholdCounts.set(2, thresholdCounts.get(2) + 1);
					}
				}
			}
		}
		
		err /= x.length;
		
		Map<String, Double> stats = new LinkedHashMap<>();
		stats.put("Error", err);
		double bestF1 = 0;
		for (Double threshold : counts.keySet()) {
			List<Integer> thresholdCounts = counts.get(threshold);
			double p = thresholdCounts.get(0) / (float) (thresholdCounts.get(0) + thresholdCounts.get(1));
			double r = thresholdCounts.get(0) / (float) (thresholdCounts.get(0) + thresholdCounts.get(3));
			double f1 = 2 * p * r / (float) (p + r);
			stats.put(String.format("P:%.2f", threshold), p);
			stats.put(String.format("R:%.2f", threshold), r);
			stats.put(String.format("F1:%.2f", threshold), Double.isNaN(f1) ? 0 : f1);
			
			bestF1 = Math.max(f1, bestF1);
		}
		stats.put("BestF1", bestF1);
		
		
		return stats;
	}
	
	private static final int MAX_ITERATIONS = 10_000;
	private static final double HOLDOUT_PERCENTAGE = 0.1;
	
	public static void main(String[] args) throws IOException {
		List<Labelset> labelSets = Data.split(new WisconsinBreastCancerData().getLabelset(), HOLDOUT_PERCENTAGE);
		Labelset trainSet = labelSets.get(0);
		Labelset testSet = labelSets.get(1);
		
		System.out.println("Training Logistic Regression using Batch Gradient Descent...");
		LogRegressionSGD model = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, true);		
		for (int i = 1; i <= MAX_ITERATIONS; i++) {
			model.train(trainSet.xs, trainSet.ys);			
		}
		System.out.println("\tTrain: "	+ getStats(model, trainSet.xs, trainSet.ys));	
		System.out.println("\tTest : " + getStats(model, testSet.xs, testSet.ys));
		
		LogRegressionSGD sgd = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS);				
		System.out.println("Training Logistic Regression using Stochastic Gradient Descent...");
		for (int i = 1; i <= MAX_ITERATIONS; i++) {
			sgd.train(trainSet.xs, trainSet.ys);			
		}
		System.out.println("\tTrain: " + getStats(sgd, trainSet.xs, trainSet.ys));	
		System.out.println("\tTest : " + getStats(sgd, testSet.xs, testSet.ys));
	}
}
