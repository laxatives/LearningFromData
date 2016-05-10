package com.diffbot.learningfromdata.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.Arrays;

public class MnistHandwrittenDigitData implements Data {
	public static final URL DATA_PATH = MnistHandwrittenDigitData.class
			.getClassLoader().getResource("mnist-digits.data");
	public static final int NUM_EXAMPLES = 42000;
	public static final int NUM_FIELDS = 784;
	public static final int SKIP_FIELDS = 1; // label
	public static final int LABEL_INDEX = 0;
	
	public Labelset getLabelset() throws IOException {
		double[][] xs = new double[NUM_EXAMPLES][NUM_FIELDS];
		double[] ys = new double[NUM_EXAMPLES];
		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(DATA_PATH.openStream()))) {
			int i = 0;
			br.readLine(); // skip header
		    for (String line; (line = br.readLine()) != null; i++) {
		    	String[] features = line.split(",");
		    	double[] x = Arrays.stream(features)
		    			.skip(SKIP_FIELDS) 
		    			.limit(NUM_FIELDS)
		    			.map(Double::valueOf)
		    			.mapToDouble(Double::doubleValue)
		    			.toArray();
		    	Double y = Double.valueOf(features[LABEL_INDEX]);
		    	
		    	xs[i] = x;
		    	ys[i] = y;
		    }
		}
		return new Labelset(xs, ys);
	}

}
