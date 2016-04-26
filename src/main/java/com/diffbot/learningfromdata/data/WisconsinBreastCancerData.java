package com.diffbot.learningfromdata.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 10 Integer valued features => Binary
 * Data is filtered on entries WITHOUT missing data.
 */
public class WisconsinBreastCancerData implements DataSet {
	public static final URL DATA_PATH = WisconsinBreastCancerData.class
			.getClassLoader().getResource("breast-cancer-wisconsin-sanitized.data");
	public static final int NUM_EXAMPLES = 682;
	public static final int NUM_FIELDS = 9;
	public static final int SKIP_FIELDS = 1; // id
	public static final int LABEL_INDEX = 10;
	public static final int LABEL_OFFSET = -3; // normalize classes from {2,4} -> {-1,+1}
	public static final Map<Integer, String> FEATURE_NAME_MAP = Arrays.stream(new Object[][]{
		{0, "Clump Thickness"},
		{1, "Uniformity of Cell Size"},
		{2, "Uniformity of Cell Shape"},
		{3, "Marginal Adhesion"},
		{4, "Single Epithelial Cell Size"},
		{5, "Bare Nuclei"},
		{6, "Bland Chromatin"},
		{7, "Normal Nucleoli"},
		{8, "Mitosis"},
		{9, "Malgignant"},
	}).collect(Collectors.toMap(kv -> (Integer) kv[0], kv -> (String) kv[1]));
	
	public TrainingExamples getTrainingExamples() throws IOException {
		double[][] xs = new double[NUM_EXAMPLES][NUM_FIELDS];
		double[] ys = new double[NUM_EXAMPLES];
		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(DATA_PATH.openStream()))) {
			int i = 0;
		    for (String line; (line = br.readLine()) != null; i++) {
		    	String[] features = line.split(",");
		    	double[] x = Arrays.stream(features)
		    			.skip(SKIP_FIELDS) 
		    			.limit(NUM_FIELDS)
		    			.map(Double::valueOf)
		    			.mapToDouble(Double::doubleValue)
		    			.toArray();
		    	Double y = Double.valueOf(features[LABEL_INDEX]) + LABEL_OFFSET;
		    	
		    	xs[i] = x;
		    	ys[i] = y;		    			
		    }
		}
		return new TrainingExamples(xs, ys);
	}
}
