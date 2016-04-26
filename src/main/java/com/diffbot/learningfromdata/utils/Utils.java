package com.diffbot.learningfromdata.utils;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Utils {

	/**
	 * Returns output compatible with Wolfram Alpha.
	 */
	public static String arrayToString(double[] ds) {
		return "{" + Arrays.stream(ds).mapToObj(d -> String.format("%05f", d)).collect(Collectors.joining(", ")) + "}"; 
	}
	
	/**
	 * Returns output compatible with Wolfram Alpha.
	 */
	public static String matrixToString(double[][] m) {
		return "{ " + Arrays.stream(m).map(row -> arrayToString(row)).collect(Collectors.joining(",\n  ")) + " }";
	}

}
