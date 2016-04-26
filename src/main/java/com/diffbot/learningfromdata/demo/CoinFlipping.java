package com.diffbot.learningfromdata.demo;

import java.util.Random;

import com.diffbot.learningfromdata.utils.PlotUtils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.graphics.Location;
import de.erichseifert.gral.plots.BarPlot;
import de.erichseifert.gral.plots.BarPlot.BarRenderer;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;

/**
 * From Exercise 1.10 in Learning from Data
 * Demonstrates the Hoeffding Inequality assumes the hypothesis is fixed
 * before data is generated/sampled.
 */
public class CoinFlipping {
	private static final double BAR_DISPLACEMENT = 1;
	private static final double BAR_WIDTH = 0.75;
	private static final double GROUP_DISPLACEMENT = 20;
	
	private static final int NUM_COINS = 1_000;
	private static final int NUM_FLIPS = 10;
	private static final int TRIALS = 10_000;
	private static final double TRUE_PROBABILITY = 0.5;
	private static final Random RANDOM = new Random();
	
	public static class ExperimentResult {
		public int first;
		public int rand;
		public int min;
		
		public ExperimentResult(int first, int rand, int min) {
			this.first = first;
			this.rand = rand;
			this.min = min;
		}
	}
	
	public static ExperimentResult runExperiment() {
		int[] coinResults  = flipCoins();
		int first = coinResults[0];
		
		int min = NUM_FLIPS;
		for (int i = 0; i < coinResults.length; i++) {
			min = coinResults[i] < min ? coinResults[i] : min;
		}
		
		int rand = coinResults[RANDOM.nextInt(NUM_COINS)];
		
		return new ExperimentResult(first, rand, min);
	}
	
	private static int[] flipCoins() {
		int[] coinResults = new int[NUM_COINS];
		for (int i = 0; i < NUM_COINS; i++) {
			int numHeads = 0;
			for (int j = 0; j < NUM_FLIPS; j++) {
				if (RANDOM.nextDouble() < TRUE_PROBABILITY) {
					numHeads += 1;
				}
			}
			coinResults[i] = numHeads;
		}
		return coinResults;
	}
	
	private static double calcHoeffdingBound(double error, int sampleSize) {
		return 2 * Math.exp(-2 * error*error * sampleSize);
	}
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		System.out.println(String.format("Running coin flipping experiment over %d trials...", TRIALS));
		int[] firstResults = new int[NUM_FLIPS + 1];
		int[] randResults = new int[NUM_FLIPS + 1];
		int[] minResults = new int[NUM_FLIPS + 1];
	
		for (int i = 1; i <= TRIALS; i++) {
			ExperimentResult result = runExperiment();
			firstResults[result.first] += 1;
			randResults[result.rand] += 1;
			minResults[result.min] += 1;
			
			if (i % (TRIALS / 10) == 0) {
				System.out.println(String.format("\tCompleted trial %d of %d.", i, TRIALS));
			}
		}
		
		DataTable results = new DataTable(Double.class, Integer.class, Double.class, 
				Integer.class, Double.class, Integer.class, Double.class);
		for (int headCount = 0; headCount <= NUM_FLIPS; headCount++) {
			double error = (double) headCount / NUM_FLIPS - TRUE_PROBABILITY;			
			double hoeffdingBound = TRIALS * calcHoeffdingBound(error, NUM_FLIPS);
			
			results.add(headCount * BAR_DISPLACEMENT, firstResults[headCount],
					headCount * BAR_DISPLACEMENT + GROUP_DISPLACEMENT, randResults[headCount], 
					headCount * BAR_DISPLACEMENT + 2 * GROUP_DISPLACEMENT, minResults[headCount],
					// this is plotted in the negative direction since gral 
					// hides smaller bins behind larger ones
					-hoeffdingBound);		
		}
		DataSeries fs = new DataSeries(results, 0, 1);
		DataSeries hfs = new DataSeries(results, 0, 6);
		DataSeries rs = new DataSeries(results, 2, 3);
		DataSeries hrs = new DataSeries(results, 2, 6);
		DataSeries ms = new DataSeries(results, 4, 5);
		DataSeries hms = new DataSeries(results, 4, 6);
		
		System.out.println("\tPlotting results...");
		BarPlot plot = new BarPlot(fs, hfs, rs, hrs, ms, hms);
		plot.getTitle().setText("Coin Flipping");
		plot.setInsets(new Insets2D.Double(20.0, 40.0, 40.0, 40.0));
		plot.setBarWidth(BAR_WIDTH);

		BarRenderer fsRenderer = new BarRenderer(plot);
		fsRenderer.setColor(PlotUtils.BLUE);
		fsRenderer.setValueVisible(true);
		plot.addPointRenderer(fs, fsRenderer);
		BarRenderer rsRenderer = new BarRenderer(plot);
		rsRenderer.setColor(PlotUtils.GOLD);
		rsRenderer.setValueVisible(true);
		plot.addPointRenderer(rs, rsRenderer);
		BarRenderer msRenderer = new BarRenderer(plot);
		msRenderer.setColor(PlotUtils.GREEN);
		msRenderer.setValueVisible(true);
		plot.addPointRenderer(ms, msRenderer);
		
		BarRenderer hsRenderer = new BarRenderer(plot);
		hsRenderer.setValueVisible(true);
		hsRenderer.setValueLocation(Location.SOUTH);
		plot.addPointRenderer(hfs,  hsRenderer);
		plot.addPointRenderer(hrs,  hsRenderer);
		plot.addPointRenderer(hms,  hsRenderer);
		LineRenderer lineRenderer = new DefaultLineRenderer2D();
		lineRenderer.setColor(PlotUtils.RED);
		plot.setLineRenderers(hfs, lineRenderer);
		plot.setLineRenderers(hrs, lineRenderer);
		plot.setLineRenderers(hms, lineRenderer);
		
		PlotUtils.drawPlot(plot);
	}
}
