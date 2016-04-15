package com.diffbot.learningfromdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import com.diffbot.learningfromdata.utils.PlotUtils;
import com.diffbot.learningfromdata.utils.Utils;
import com.diffbot.learningfromdata.utils.Utils.TrainingExamples;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.graphics.Location;
import de.erichseifert.gral.plots.BarPlot;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.BarPlot.BarRenderer;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;

public class BreastCancerPerceptronModel {
	private static final String DATA_DIR = "/home/bhan/workspace/LearningFromData/src/main/resources";
	// removed all examples missing data
	private static final File DATA_FILE = new File(DATA_DIR, "breast-cancer-wisconsin-sanitized.data");
	private static final int NUM_EXAMPLES = 682;
	private static final int NUM_FIELDS = 9;
	private static final int LABEL_INDEX = 10;
	private static final int LABEL_OFFSET = -3; // normalize classes from {2,4} -> {-1,+1}
	private static final int MAX_ITERATIONS = 100;
	// cat /home/bhan/workspace/LearningFromData/src/main/resources/breast-cancer-wisconsin-sanitized.data | cut -d ',' -f 10 | sort | uniq -c | sort -n
	private static final Map<Integer, String> featureNameMap = Arrays.stream(new Object[][]{
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
	
	public static void main(String[] args) throws IOException {
		TrainingExamples es = getTrainingExamples(DATA_FILE);
		PerceptronClassifier classifier = new PerceptronClassifier(NUM_FIELDS);
		
		System.out.println("Training perceptron using Wisconsin Breast Cancer dataset...");
		classifier.train(es.xs,  es.ys, MAX_ITERATIONS, true);
		plot(es);
		
		printStats(classifier, es);
	}
	
	private static TrainingExamples getTrainingExamples(File dataFile) throws IOException {
		double[][] xs = new double[NUM_EXAMPLES][NUM_FIELDS];
		double[] ys = new double[NUM_EXAMPLES];
		try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
			int i = 0;
		    for (String line; (line = br.readLine()) != null; i++) {
		    	String[] features = line.split(",");
		    	double[] x = Arrays.stream(features).skip(1).limit(NUM_FIELDS).map(Double::valueOf).mapToDouble(Double::doubleValue).toArray();
		    	Double y = Double.valueOf(features[LABEL_INDEX]) + LABEL_OFFSET;
		    	
		    	xs[i] = x;
		    	ys[i] = y;		    			
		    }
		}
		return new TrainingExamples(xs, ys);
	}
	
	@SuppressWarnings("unchecked")
	private static void plot(TrainingExamples es) {
		// plot against each feature
		for (int plotInd = 0; plotInd <= NUM_FIELDS; plotInd ++) {
			System.out.println("\tPlotting results for input feature index " + plotInd + "...");		
			Map<Integer, Integer> t_h = new HashMap<>();
			Map<Integer, Integer> f_h = new HashMap<>();
			for (int i = 0; i < es.xs.length; i++) {
				double[] x = es.xs[i];
				double y = es.ys[i];
				int pltVal = plotInd < NUM_FIELDS ? (int) x[plotInd] : (int) y;
				if (y > 0) {
					t_h.put(pltVal, t_h.getOrDefault(pltVal, 0) + 1);
				} else {
					f_h.put(pltVal, f_h.getOrDefault(pltVal, 0) + 1);
				}
			}

			DataTable t_t = new DataTable(Integer.class, Integer.class);
			for (Entry<Integer, Integer> e : t_h.entrySet()) {
				t_t.add(e.getKey(), e.getValue());
			}
		
			DataTable f_t = new DataTable(Integer.class, Integer.class);
			for (Entry<Integer, Integer> e : f_h.entrySet()) {
				f_t.add(e.getKey(), -e.getValue());
			}

			DataSeries t_s = new DataSeries("True", t_t, 0, 1);
			DataSeries f_s = new DataSeries("False", f_t, 0, 1);
			BarPlot plot = new BarPlot(t_s, f_s);
		
			PointRenderer t_r = new BarRenderer(plot);
			t_r.setColor(PlotUtils.BLUE);
			t_r.setValueVisible(true);		
			plot.setPointRenderers(t_s, t_r);
			PointRenderer f_r = new BarRenderer(plot);
			f_r.setColor(PlotUtils.GOLD);
			f_r.setValueVisible(true);		
			f_r.setValueLocation(Location.SOUTH);
			plot.setPointRenderers(f_s, f_r);		
	
			plot.setInsets(new Insets2D.Double(20.0, 40.0, 40.0, 40.0));
			plot.getTitle().setText("Feature \"" + featureNameMap.get(plotInd) + "\" Histogram");	
			
			PlotUtils.drawPlot(plot);
		}	
	}
	
	private static void printStats(PerceptronClassifier classifier, TrainingExamples es) {
		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;
		for (int i = 0; i < es.xs.length; i++) {
			double[] x = es.xs[i];
			boolean y = es.ys[i] > 0;
			boolean guess = classifier.classify(x) > 0;
			if (y) {
				if (guess) {
					tp += 1;
				} else {
					fn += 1;
				}
			} else {
				if (guess) {
					fp += 1;
				} else {
					tn += 1;
				}
			}
		}
		System.out.println();
		System.out.println("    Confusion   ||  Predicted");
		System.out.println("       Matrix   ||    T   |  F");
		System.out.println("---------------------------------");
		System.out.println(String.format("   Actual |  T  ||   %d  |    %d", tp, fn));
		System.out.println(String.format("          |  F  ||     %d  |  %d", fp, tn));
	}

}