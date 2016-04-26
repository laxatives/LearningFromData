package com.diffbot.learningfromdata.demo;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.diffbot.learningfromdata.classifiers.PerceptronClassifier;
import com.diffbot.learningfromdata.data.DataSet.TrainingExamples;
import com.diffbot.learningfromdata.data.WisconsinBreastCancerData;
import com.diffbot.learningfromdata.utils.PlotUtils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.graphics.Location;
import de.erichseifert.gral.plots.BarPlot;
import de.erichseifert.gral.plots.BarPlot.BarRenderer;
import de.erichseifert.gral.plots.points.PointRenderer;

/**
 * Applies Pocket Perceptron Learning Algorithm to Wisconsin Breast Cancer data
 * and plots histograms for each feature.
 */
public class BreastCancerPerceptronModel {
	public static final int MAX_ITERATIONS = 100;
	private static final int NUM_FIELDS = WisconsinBreastCancerData.NUM_FIELDS;
	
	public static void main(String[] args) throws IOException {
		TrainingExamples es = new WisconsinBreastCancerData().getTrainingExamples();
		PerceptronClassifier classifier = new PerceptronClassifier(NUM_FIELDS);
		
		System.out.println("Training perceptron using Wisconsin Breast Cancer dataset...");
		classifier.train(es.xs,  es.ys, MAX_ITERATIONS, true);
		plot(es);
		
		classifier.printStats(es);
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
			plot.getTitle().setText("Feature \"" + WisconsinBreastCancerData.FEATURE_NAME_MAP.get(plotInd) + "\" Histogram");	
			
			PlotUtils.drawPlot(plot);
		}	
	}
}