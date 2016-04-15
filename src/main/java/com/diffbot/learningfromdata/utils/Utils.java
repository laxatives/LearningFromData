package com.diffbot.learningfromdata.utils;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Utils {

	public static String arrayToString(double[] ds) {
		return "{" + Arrays.stream(ds).mapToObj(d -> String.format("%05f", d)).collect(Collectors.joining(", ")) + "}"; 
	}
	
	public static String matrixToString(double[][] m) {
		return "{ " + Arrays.stream(m).map(row -> arrayToString(row)).collect(Collectors.joining(",\n  ")) + " }";
	}

	public static class TrainingExamples {
		public double[][] xs;
		public double[] ys;
		public TrainingExamples(double[][] xs, double[] ys) {
			this.xs = xs;
			this.ys = ys;
		}
	}
}
