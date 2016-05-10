package com.diffbot.learningfromdata.demo;

import java.util.Arrays;
import java.util.Random;

import com.diffbot.learningfromdata.classifiers.PerceptronClassifier;
import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.PlotUtils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;

/**
 * From exercise 3.2 in Learning From Data
 * Runs pocket PLA against inseparable dataset.
 */

public class PocketPLADemo {
	private static final int DOF = 16;
	private static final int NUM_EXAMPLES = 1024;
	private static final int TEST_EXAMPLES = 1_000;
	private static final int NUM_ITERATIONS = 1_000;
	private static final int NUM_EXPERIMENTS = 8;
	private static Random RANDOM = new Random();
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {		
		int[][] inErrors = new int[NUM_ITERATIONS][NUM_EXPERIMENTS];
		int[][] pocketInErrors = new int[NUM_ITERATIONS][NUM_EXPERIMENTS];
		int[][] outErrors = new int[NUM_ITERATIONS][NUM_EXPERIMENTS];
		int[][] pocketOutErrors = new int[NUM_ITERATIONS][NUM_EXPERIMENTS];
		
		PerceptronClassifier perceptron = new PerceptronClassifier(DOF);
		for (int i = 0; i < NUM_EXPERIMENTS; i++) {					
			double trueBias = RANDOM.nextDouble();
			double[] trueWeights = new double[DOF + 1];
			for (int j = 0; j < trueWeights.length; j++) {
				trueWeights[j] = RANDOM.nextDouble();
			}
			
			double[][] trainXs = new double[NUM_EXAMPLES][DOF];
			double[] trainYs = new double[NUM_EXAMPLES];
			for (int j = 0; j < NUM_EXAMPLES; j++) {
				for (int k = 0; k < DOF; k++) {
					trainXs[j][k] = RANDOM.nextDouble();
				}
				trainYs[j] = (MathUtils.dotProduct(trainXs[j], trueWeights) > trueBias) ? 1 : -1;
				if (RANDOM.nextDouble() < 0.1) {
					trainYs[j] = -trainYs[j];
				}
			}
			
			double[][] testXs = new double[TEST_EXAMPLES][DOF];
			double[] testYs = new double[TEST_EXAMPLES];
			for (int j = 0; j < TEST_EXAMPLES; j++) {
				for (int k = 0; k < DOF; k++) {
					testXs[j][k] = RANDOM.nextDouble();
				}
				testYs[j] = (MathUtils.dotProduct(testXs[j], trueWeights) > trueBias) ? 1 : -1;
			}
			
			for (int t = 0; t < NUM_ITERATIONS; t++) {
				perceptron.train(trainXs, trainYs);
				
				int inError = 0;
				int pocketInError = 0;
				for (int j = 0; j < NUM_EXAMPLES; j++) {
					double estimate = perceptron.classify(trainXs[j]);
					if (isMislabeled(estimate, trainYs[i])) inError++;
					double pocketEstimate = perceptron.classify(trainXs[j], true);
					if (isMislabeled(pocketEstimate, trainYs[i])) pocketInError++;
				}
				inErrors[t][i] = inError;
				pocketInErrors[t][i] = pocketInError;
				
				int outError = 0;
				int pocketOutError = 0;
				for (int j = 0; j < TEST_EXAMPLES; j++) {
					double estimate = perceptron.classify(testXs[j]);
					if (isMislabeled(estimate, testYs[i])) outError++;
					double pocketEstimate = perceptron.classify(testXs[j], true);
					if (isMislabeled(pocketEstimate, testYs[i])) pocketOutError++;
				}
				outErrors[t][i] = outError;
				pocketOutErrors[t][i] = pocketOutError;				
			}
			perceptron = new PerceptronClassifier(DOF);
		}		
		
		DataTable inErrDt = new DataTable(Integer.class, Double.class);
		DataTable pocketInErrDt = new DataTable(Integer.class, Double.class);
		DataTable outErrDt = new DataTable(Integer.class, Double.class);
		DataTable pocketOutErrDt = new DataTable(Integer.class, Double.class);
		
		for (int t = 0; t < NUM_ITERATIONS; t++) {
			inErrDt.add(t, Arrays.stream(inErrors[t]).sum() / (double) NUM_EXPERIMENTS);
			pocketInErrDt.add(t, Arrays.stream(pocketInErrors[t]).sum() / (double) NUM_EXPERIMENTS);
			outErrDt.add(t, Arrays.stream(outErrors[t]).sum() / (double) NUM_EXPERIMENTS);
			pocketOutErrDt.add(t, Arrays.stream(pocketOutErrors[t]).sum() / (double) NUM_EXPERIMENTS);
		}
		
		DataSeries tpD2s = new DataSeries("In Error", inErrDt, 0, 1);
		DataSeries fpD2s = new DataSeries("Pocket In Error", pocketInErrDt, 0, 1);
		DataSeries tnD2s = new DataSeries("Out Error", outErrDt, 0, 1);
		DataSeries fnD2s = new DataSeries("Pocket Out Error", pocketOutErrDt, 0, 1);
		XYPlot plot = new XYPlot(tpD2s, fpD2s, tnD2s, fnD2s);
		
		PointRenderer tpRenderer = new DefaultPointRenderer2D();
		tpRenderer.setColor(PlotUtils.BLUE);
		plot.setPointRenderers(tpD2s, tpRenderer);
		PointRenderer tnRenderer = new DefaultPointRenderer2D();
		tnRenderer.setColor(PlotUtils.GOLD);
		plot.setPointRenderers(tnD2s, tnRenderer);		
		PointRenderer fpRenderer = new DefaultPointRenderer2D();
		fpRenderer.setColor(PlotUtils.GREEN);
		plot.setPointRenderers(fpD2s, fpRenderer);
		PointRenderer fnRenderer = new DefaultPointRenderer2D();
		fnRenderer.setColor(PlotUtils.RED);
		plot.setPointRenderers(fnD2s, fnRenderer);

		plot.setInsets(new Insets2D.Double(20.0, 40.0, 40.0, 40.0));
		plot.getTitle().setText("Classic vs Pocket PLA Errors over time");	
		plot.setLegendVisible(true);
		
		PlotUtils.drawPlot(plot);
	}
	
	private static boolean isMislabeled(double estimate, double y) {
		return (estimate > 0 && y < 0) || (estimate < 0 && y > 0);
	}
}
