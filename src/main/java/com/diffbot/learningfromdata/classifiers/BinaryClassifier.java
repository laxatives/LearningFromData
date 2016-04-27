package com.diffbot.learningfromdata.classifiers;

import com.diffbot.learningfromdata.data.Data.Labelset;

public interface BinaryClassifier {

	public int train(double[][] x, double[] y);	
	public double classify(double[] input);
	
	public default void printStats(Labelset es) {
		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;
		for (int i = 0; i < es.xs.length; i++) {
			double[] x = es.xs[i];
			boolean y = es.ys[i] > 0;
			boolean h = this.classify(x) > 0;
			if (y) {
				if (h) {
					tp += 1;
				} else {
					fn += 1;
				}
			} else {
				if (h) {
					fp += 1;
				} else {
					tn += 1;
				}
			}
		}
		
		System.out.println();
		System.out.println("    Confusion   ||   Predicted");
		System.out.println("       Matrix   ||    T   |  F");
		System.out.println("---------------------------------");
		System.out.println(String.format("   Actual |  T  ||  %4d  |  %4d", tp, fn));
		System.out.println(String.format("          |  F  ||  %4d  |  %4d", fp, tn));		
		System.out.println("TOT: " + (tp + fp + tn + fn));
		float p = tp / (float) (tp + fp);
		float r = tp / (float) (tp + fn);
		System.out.println("P/R: " + p + "/" + r);
		System.out.println("F1: " + 2 * p * r / (p + r));
	}
}
