package com.diffbot.learningfromdata.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.google.common.primitives.Doubles;

public interface Data {

	Labelset getLabelset() throws IOException;	

	public static class Labelset {
		public double[][] xs;
		public double[] ys;
		public Labelset(double[][] xs, double[] ys) {
			this.xs = xs;
			this.ys = ys;
		}
	}
	
	public static List<Labelset> split(Labelset es, double holdoutPct) {
		Random random = new Random();
		List<double[]> trainingXs = new ArrayList<>();
		List<Double> trainingYs = new ArrayList<>();
		List<double[]> holdoutXs = new ArrayList<>();
		List<Double> holdoutYs = new ArrayList<>();
		for (int i = 0; i < es.xs.length; i++) {
			if (random.nextDouble() > holdoutPct) {
				trainingXs.add(es.xs[i]);
				trainingYs.add(es.ys[i]);
			} else {
				holdoutXs.add(es.xs[i]);
				holdoutYs.add(es.ys[i]);				
			}
		}
		
		Labelset trainingLabels = new Labelset(trainingXs.toArray(new double[][]{}), Doubles.toArray(trainingYs));
		Labelset holdoutLabels = new Labelset(holdoutXs.toArray(new double[][]{}), Doubles.toArray(holdoutYs));
		return Arrays.asList(trainingLabels, holdoutLabels);
	}
}