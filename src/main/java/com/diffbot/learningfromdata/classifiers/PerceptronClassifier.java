package com.diffbot.learningfromdata.classifiers;

import java.util.Random;

import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.PlotUtils;
import com.diffbot.learningfromdata.utils.Utils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;

public class PerceptronClassifier implements BinaryClassifier {
	private static boolean POCKET = true;
	
	public double[] weights;
	public double bias;

	public PerceptronClassifier(int numFeatures) {
		weights = new double[numFeatures];
		bias = 0;
	}
	
	/**
	 * Performs one epoch on the given dataset, updating weights as necessary.
	 * Returns the number of mislabeled elements.
	 */
	public int train(double[][] x, double[] y) {
		int numMislabeled = 0;
		for (int i = 0; i < x.length; i++) {
			boolean correctlyLabeled = update(x[i], y[i]);
			if (!correctlyLabeled) {
				numMislabeled += 1;
			}
		}
		return numMislabeled;
	}
	
	public int train(double[][] x, double[] y, int numIterations, boolean log) {
		long start = System.currentTimeMillis();
		int errors = 0;
		for (int i = 1; i <= numIterations; i++) {
			errors = train(x, y); 
			if (errors == 0) {
				if (log) {
					System.out.println(String.format("\tPerceptron converged in %d iterations.", i));
				}
				break;
			}
			if (log) {
				System.out.println(String.format("\tCompleted epoch %d with %d mislabeled out of %d total examples.", i, errors, y.length));
				System.out.println(String.format("\tEstimated Weights: %s\n\t\tBias: %.2f", Utils.arrayToString(weights), bias));
			}
		}
		float took = System.currentTimeMillis() - start;
		System.out.println(String.format("\tTook %f ms (%.10f per instance)", took, took / y.length));
		return errors;
	}
	
	/**
	 * Classifies the training example, updating weights as necessary.
	 * Returns true iff the input was correctly classified.
	 */
	private boolean update(double[] x, double y) {		
		double h = classify(x);
		if (h == y) {
			return true;
		}
		
		double[] grad = MathUtils.scalarProduct(y, x); 
		weights = MathUtils.sumArrays(weights, grad);
		bias -= y;
		
		return false;
	}
	
	public double classify(double[] input) {
		return MathUtils.dotProduct(input, weights) > bias ? 1 : -1; 
	}	
	
	private static final Random RANDOM = new Random();	
	private static final boolean LINEARLY_SEPARABLE = false;
	private static final int NUM_FEATURES = 2;
	private static final int MAX_EPOCHS = 10_000;
	private static final int SAMPLE_COUNT = 8_000;
	private static final int HOLDOUT_COUNT = SAMPLE_COUNT / 2;
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		System.out.println(String.format("Training perceptron using degree 2 feature set and %d...", SAMPLE_COUNT));
		double[] trueWeights = new double[] {RANDOM.nextGaussian(), RANDOM.nextGaussian()};
		double trueBias = RANDOM.nextGaussian();
		
		System.out.println("\tGenerating dataset...");
		double[][] trainingSet = new double[SAMPLE_COUNT][NUM_FEATURES];
		double[] labels = new double[SAMPLE_COUNT];
		double[][] holdoutSet = new double[HOLDOUT_COUNT][NUM_FEATURES];
		double[] holdoutLabels = new double[HOLDOUT_COUNT];
		if (LINEARLY_SEPARABLE) {
			System.out.println(String.format("\tTrue weight vector: %s\n\tTrue bias: ", Utils.arrayToString(trueWeights), trueBias));
			for (int i = 0; i < trainingSet.length; i++) {
				for (int j = 0; j < NUM_FEATURES; j++) {
					trainingSet[i][j] = 10*RANDOM.nextGaussian();
				}
				labels[i] = MathUtils.dotProduct(trueWeights, trainingSet[i]) > trueBias ? 1 : -1;
			}
			for (int i = 0; i < holdoutSet.length; i++) {
				for (int j = 0; j < NUM_FEATURES; j++) {
					holdoutSet[i][j] = 10*RANDOM.nextGaussian();
				}
				holdoutLabels[i] = MathUtils.dotProduct(trueWeights, holdoutSet[i]) > trueBias ? 1 : -1;
			}
		} else {
			for (int i = 0; i < trainingSet.length; i++) {
				boolean label = RANDOM.nextBoolean();
				for (int j = 0; j < NUM_FEATURES; j++) {
					trainingSet[i][j] = 10*RANDOM.nextGaussian() + (label ? 15 : -15);
				}
				labels[i] = label ? 1 : -1;
			}
			for(int i = 0; i < holdoutSet.length; i++){
				boolean label = RANDOM.nextBoolean();
				for (int j = 0; j < NUM_FEATURES; j++) {
					holdoutSet[i][j] = 10*RANDOM.nextGaussian() + (label ? 15 : -15);
				}
				holdoutLabels[i] = label ? 1 : -1;
			}
		}
		
		System.out.println("\tTraining degree 2 perceptron...");
		long start = System.currentTimeMillis();
		PerceptronClassifier perceptron = new PerceptronClassifier(NUM_FEATURES);
		int prevErrors = trainingSet.length;
		double[] prevWeights = perceptron.weights;
		for (int i = 1; i <= MAX_EPOCHS; i++) {
			int errors = perceptron.train(trainingSet, labels); 
			if (errors == 0) {
				System.out.println(String.format("\t\tPerceptron converged in %d iterations.", i));
				break;
			}
			
			System.out.println(String.format("\t\tCompleted epoch %d with %d mislabeled out of %d total examples.", i, errors, trainingSet.length));
			System.out.println(String.format("\t\tWeights: %s\n\t\tBias: %.2f", Utils.arrayToString(perceptron.weights), perceptron.bias));
			
			if (POCKET && errors > prevErrors) {
				System.out.println(String.format("\t\tPocket perceptron performance peaked in %d iterations with %d errors", i, errors));
				perceptron.weights = prevWeights;
				break;
			}
			
			prevErrors = errors;
			prevWeights = perceptron.weights;
		}

		System.out.println(String.format("\tTrue weight vector: %s\n\tTrue bias: %.2f", Utils.arrayToString(trueWeights), trueBias));
		System.out.println(String.format("\tEstimated weight vector: %s\n\tEstimated bias: %.2f", Utils.arrayToString(perceptron.weights), perceptron.bias));
		
		float took = System.currentTimeMillis() - start;
		System.out.println(String.format("\tTook %f ms (%.10f per instance)", took, took / SAMPLE_COUNT));
				
		System.out.println("\tPlotting results...");
		DataTable truePositivesD2 = new DataTable(Double.class, Double.class);
		DataTable falsePositivesD2 = new DataTable(Double.class, Double.class);
		DataTable trueNegativesD2 = new DataTable(Double.class, Double.class);
		DataTable falseNegativesD2 = new DataTable(Double.class, Double.class);
		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;
		for (int i = 0; i < holdoutSet.length; i++) {
			double[] input = holdoutSet[i];
			double label = holdoutLabels[i];
			if (label > 0) {
				if (perceptron.classify(input) > 0) {
					truePositivesD2.add(input[0], input[1]);
					tp++;
				} else {
					falseNegativesD2.add(input[0], input[1]);
					fn++;
				}
			} else {
				if (perceptron.classify(input) > 0) {
					falsePositivesD2.add(input[0], input[1]);
					fp++;
				} else {
					trueNegativesD2.add(input[0], input[1]);
					tn++;
				}				
			}
		}
		
		System.out.println("TP: " + tp);
		System.out.println("FP: " + fp);
		System.out.println("TN: " + tn);
		System.out.println("FN: " + fn);
		System.out.println("TOT: " + holdoutLabels.length);
		float p = tp / (float) (tp + fp);
		float r = tp / (float) (tp + fn);
		System.out.println("P/R: " + p + "/" + r);
		System.out.println("F1: " + 2 * p * r / (p + r));

		DataSeries tpD2s = new DataSeries("True positives", truePositivesD2, 0, 1);
		DataSeries fpD2s = new DataSeries("False positives", falsePositivesD2, 0, 1);
		DataSeries tnD2s = new DataSeries("True negatives", trueNegativesD2, 0, 1);
		DataSeries fnD2s = new DataSeries("False negatives", falseNegativesD2, 0, 1);
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
		plot.getTitle().setText("Perceptron using generated dataset");	
		plot.setLegendVisible(true);
	
		PlotUtils.drawPlot(plot);
	} 
	
}
