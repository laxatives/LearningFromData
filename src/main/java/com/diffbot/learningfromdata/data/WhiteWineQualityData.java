package com.diffbot.learningfromdata.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.Arrays;

public class WhiteWineQualityData implements DataSet {
	public static final URL DATA_PATH = WisconsinBreastCancerData.class
			.getClassLoader().getResource("winequality-white.csv");
	public static final int NUM_EXAMPLES = 4898;
	public static final int NUM_FIELDS = 11;
	
	public TrainingExamples getTrainingExamples() throws IOException {
		double[][] xs = new double[NUM_EXAMPLES][NUM_FIELDS];
		double[] ys = new double[NUM_EXAMPLES];
		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(DATA_PATH.openStream()))) {
			int i = 0;
			String line = br.readLine(); // skip header
		    while ((line = br.readLine()) != null) {
		    	String[] features = line.split(";");
		    	double[] x = Arrays.stream(features)
		    			.limit(NUM_FIELDS)
		    			.map(Double::valueOf)
		    			.mapToDouble(Double::doubleValue)
		    			.toArray();
		    	Double y = Double.valueOf(features[NUM_FIELDS]);
		    	
		    	xs[i] = x;
		    	ys[i] = y;
		    	i++;
		    }
		}
		return new TrainingExamples(xs, ys);
	}
}
