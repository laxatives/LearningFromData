package com.diffbot.learningfromdata.demo;

import java.awt.Color;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.diffbot.learningfromdata.data.Data;
import com.diffbot.learningfromdata.data.WisconsinBreastCancerData;
import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.regression.LogRegressionGD;
import com.diffbot.learningfromdata.regression.LogRegressionGD.Variant;
import com.diffbot.learningfromdata.utils.PlotUtils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;

public class GradientDescentDemo {
	
	private static final int NUM_EXPERIMENTS = 5; 
	private static final int MAX_ITERATIONS = 4_000;
	private static final double HOLDOUT_PERCENTAGE = 0.1;

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException {
		// TODO: plot against time
		List<List<Double>> batchErrors = new ArrayList<>();
		List<List<Double>> stochasticErrors = new ArrayList<>();
		List<List<Double>> momentumErrors = new ArrayList<>();
		List<List<Double>> conjugateErrors = new ArrayList<>();
		List<List<Double>> newtonErrors = new ArrayList<>();

		List<List<Long>> batchTimes = new ArrayList<>();
		List<List<Long>> stochasticTimes = new ArrayList<>();
		List<List<Long>> momentumTimes = new ArrayList<>();
		List<List<Long>> conjugateTimes = new ArrayList<>();
		List<List<Long>> newtonTimes = new ArrayList<>();
		for (int i = 0; i < MAX_ITERATIONS; i++) {
			batchErrors.add(new ArrayList<>());
			stochasticErrors.add(new ArrayList<>());
			momentumErrors.add(new ArrayList<>());
			conjugateErrors.add(new ArrayList<>());
			newtonErrors.add(new ArrayList<>());
			
			batchTimes.add(new ArrayList<>());
			stochasticTimes.add(new ArrayList<>());
			momentumTimes.add(new ArrayList<>());
			conjugateTimes.add(new ArrayList<>());
			newtonTimes.add(new ArrayList<>());
		}

		for (int i = 1; i <= NUM_EXPERIMENTS; i++) {
			System.out.println("Beginning experiment " + i + " of " + NUM_EXPERIMENTS + "...");
			List<Labelset> labelSets = Data.split(new WisconsinBreastCancerData().getLabelset(), HOLDOUT_PERCENTAGE);
			Labelset trainSet = labelSets.get(0);
			Labelset testSet = labelSets.get(1);
			
			System.out.println("\tTraining Logistic Regression using Batch Gradient Descent...");
			LogRegressionGD batch = new LogRegressionGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.BATCH);
			long start = System.currentTimeMillis();
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				batch.train(trainSet.xs, trainSet.ys);			
				batchErrors.get(j).add(LogRegressionGD.getError(batch, testSet.xs, testSet.ys));
				batchTimes.get(j).add(System.currentTimeMillis() - start);
			}
			
			System.out.println("\tTraining Logistic Regression using Stochastic Gradient Descent...");
			LogRegressionGD sgd = new LogRegressionGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.STOCHASTIC);
			start = System.currentTimeMillis();
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				sgd.train(trainSet.xs, trainSet.ys);			
				stochasticErrors.get(j).add(LogRegressionGD.getError(sgd, testSet.xs, testSet.ys));
				stochasticTimes.get(j).add(System.currentTimeMillis() - start);
			}		
		
			System.out.println("\tTraining Logistic Regression using Momentum Gradient Descent...");
			LogRegressionGD momentum = new LogRegressionGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.MOMENTUM);
			start = System.currentTimeMillis();
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				momentum.train(trainSet.xs, trainSet.ys);			
				momentumErrors.get(j).add(LogRegressionGD.getError(momentum, testSet.xs, testSet.ys));
				momentumTimes.get(j).add(System.currentTimeMillis() - start);
			}
		
			System.out.println("\tTraining Logistic Regression using Conjugate Gradient Descent...");
			LogRegressionGD conjugate = new LogRegressionGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.CONJUGATE);
			start = System.currentTimeMillis();
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				conjugate.train(trainSet.xs, trainSet.ys);			
				conjugateErrors.get(j).add(LogRegressionGD.getError(conjugate, testSet.xs, testSet.ys));
				conjugateTimes.get(j).add(System.currentTimeMillis() - start);
			}
		
			System.out.println("\tTraining Logistic Regression using Newton's Method Gradient Descent...");
			LogRegressionGD newton = new LogRegressionGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.CONJUGATE);
			start = System.currentTimeMillis();
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				newton.train(trainSet.xs, trainSet.ys);			
				newtonErrors.get(j).add(LogRegressionGD.getError(newton, testSet.xs, testSet.ys));
				newtonTimes.get(j).add(System.currentTimeMillis() - start);
			}		
		}

		DataTable batchData = new DataTable(Integer.class, Double.class);
		DataTable stochasticData = new DataTable(Integer.class, Double.class);
		DataTable momentumData = new DataTable(Integer.class, Double.class);
		DataTable conjugateData = new DataTable(Integer.class, Double.class);
		DataTable newtonData = new DataTable(Integer.class, Double.class);
		
		DataTable batchTimingData = new DataTable(Double.class, Double.class);
		DataTable stochasticTimingData = new DataTable(Double.class, Double.class);
		DataTable momentumTimingData = new DataTable(Double.class, Double.class);
		DataTable conjugateTimingData = new DataTable(Double.class, Double.class);
		DataTable newtonTimingData = new DataTable(Double.class, Double.class);
		for (int i = 0; i < MAX_ITERATIONS; i++) {
			double batchErr = batchErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble();
			batchData.add(i, batchErr);
			batchTimingData.add(batchTimes.get(i).stream().mapToDouble(d -> d).average().getAsDouble(), batchErr);
			
			double stochasticErr = stochasticErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble();
			stochasticData.add(i, stochasticErr);
			stochasticTimingData.add(stochasticTimes.get(i).stream().mapToDouble(d -> d).average().getAsDouble(), stochasticErr);			
			
			double momentumErr = momentumErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble();
			momentumData.add(i, momentumErr);
			momentumTimingData.add(momentumTimes.get(i).stream().mapToDouble(d -> d).average().getAsDouble(), momentumErr);
			
			double conjugateErr = conjugateErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble();
			conjugateData.add(i, conjugateErr);
			conjugateTimingData.add(conjugateTimes.get(i).stream().mapToDouble(d -> d).average().getAsDouble(), conjugateErr);
			
			double newtonErr = newtonErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble();
			newtonData.add(i, newtonErr);
			newtonTimingData.add(newtonTimes.get(i).stream().mapToDouble(d -> d).average().getAsDouble(), newtonErr);
		}

		// plot error vs iteration
		DataSeries batchDS = new DataSeries("Batch", batchData, 0, 1);
		DataSeries stochasticDS = new DataSeries("Stochastic", stochasticData, 0, 1);
		DataSeries momentumDS = new DataSeries("Momentum", momentumData, 0, 1);
		DataSeries conjugateDS = new DataSeries("Conjugate", conjugateData, 0, 1);
		DataSeries newtonDS = new DataSeries("Conjugate", conjugateData, 0, 1);
		XYPlot plot = new XYPlot(batchDS, stochasticDS, momentumDS, conjugateDS /*, newtonDS*/);
		
		setLineRenderer(plot, batchDS, PlotUtils.BLUE);
		setLineRenderer(plot, stochasticDS, PlotUtils.GOLD);
		setLineRenderer(plot, momentumDS, PlotUtils.RED);
		setLineRenderer(plot, conjugateDS, PlotUtils.GREEN);
		setLineRenderer(plot, newtonDS, Color.ORANGE);
		
		plot.setInsets(new Insets2D.Double(20.0, 40.0, 40.0, 40.0));
		plot.getTitle().setText("Gradient Descent Algorithms: Error over # Iterations");	
		plot.setLegendVisible(true);
	
		PlotUtils.drawPlot(plot);
		
		// plot error vs time
		DataSeries batchTimeDS = new DataSeries("Batch", batchTimingData, 0, 1);
		DataSeries stochasticTimeDS = new DataSeries("Stochastic", stochasticTimingData, 0, 1);
		DataSeries momentumTimeDS = new DataSeries("Momentum", momentumTimingData, 0, 1);
		DataSeries conjugateTimeDS = new DataSeries("Conjugate", conjugateTimingData, 0, 1);
		DataSeries newtonTimeDS = new DataSeries("Conjugate", conjugateTimingData, 0, 1);
		XYPlot timePlot = new XYPlot(batchTimeDS, stochasticTimeDS, momentumTimeDS, conjugateTimeDS /*, newtonDS*/);
		
		setLineRenderer(timePlot, batchTimeDS, PlotUtils.BLUE);
		setLineRenderer(timePlot, stochasticTimeDS, PlotUtils.GOLD);
		setLineRenderer(timePlot, momentumTimeDS, PlotUtils.RED);
		setLineRenderer(timePlot, conjugateTimeDS, PlotUtils.GREEN);
		setLineRenderer(timePlot, newtonTimeDS, Color.ORANGE);
		
		timePlot.setInsets(new Insets2D.Double(20.0, 40.0, 40.0, 40.0));
		timePlot.getTitle().setText("Gradient Descent Algorithms: Error over Time (ms)");	
		timePlot.setLegendVisible(true);
	
		PlotUtils.drawPlot(timePlot);

	}

	private static void setLineRenderer(XYPlot plot, DataSeries ds, Color color) {
		LineRenderer lr = new DefaultLineRenderer2D();
		lr.setColor(color);
		plot.setLineRenderers(ds, lr);
		plot.setPointRenderers(ds, null);
	}
}
