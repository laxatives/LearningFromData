package com.diffbot.learningfromdata.demo;

import java.awt.Color;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.diffbot.learningfromdata.data.Data;
import com.diffbot.learningfromdata.data.WisconsinBreastCancerData;
import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.regression.LogRegressionSGD;
import com.diffbot.learningfromdata.regression.LogRegressionSGD.Variant;
import com.diffbot.learningfromdata.utils.PlotUtils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;

public class GradientDescentDemo {
	
	private static final int NUM_EXPERIMENTS = 10; // 10
	private static final int MAX_ITERATIONS = 10_000;
	private static final double HOLDOUT_PERCENTAGE = 0.1;

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException {
		// TODO: plot against time
		List<List<Double>> batchErrors = new ArrayList<>();
		List<List<Double>> stochasticErrors = new ArrayList<>();
		List<List<Double>> momentumErrors = new ArrayList<>();
		List<List<Double>> conjugateErrors = new ArrayList<>();
		List<List<Double>> newtonErrors = new ArrayList<>();
		for (int i = 0; i < MAX_ITERATIONS; i++) {
			batchErrors.add(new ArrayList<>());
			stochasticErrors.add(new ArrayList<>());
			momentumErrors.add(new ArrayList<>());
			conjugateErrors.add(new ArrayList<>());
			newtonErrors.add(new ArrayList<>());
		}

		for (int i = 1; i <= NUM_EXPERIMENTS; i++) {
			System.out.println("Beginning experiment " + i + " of " + NUM_EXPERIMENTS + "...");
			List<Labelset> labelSets = Data.split(new WisconsinBreastCancerData().getLabelset(), HOLDOUT_PERCENTAGE);
			Labelset trainSet = labelSets.get(0);
			Labelset testSet = labelSets.get(1);
			
			System.out.println("\tTraining Logistic Regression using Batch Gradient Descent...");
			LogRegressionSGD batch = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.BATCH);		
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				batch.train(trainSet.xs, trainSet.ys);			
				batchErrors.get(j).add(LogRegressionSGD.getError(batch, testSet.xs, testSet.ys));
			}
			
			System.out.println("\tTraining Logistic Regression using Stochastic Gradient Descent...");
			LogRegressionSGD sgd = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.STOCHASTIC);				
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				sgd.train(trainSet.xs, trainSet.ys);			
				stochasticErrors.get(j).add(LogRegressionSGD.getError(sgd, testSet.xs, testSet.ys));
			}		
		
			System.out.println("\tTraining Logistic Regression using Momentum Gradient Descent...");
			LogRegressionSGD momentum = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.MOMENTUM);		
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				momentum.train(trainSet.xs, trainSet.ys);			
				momentumErrors.get(j).add(LogRegressionSGD.getError(momentum, testSet.xs, testSet.ys));
			}
		
			System.out.println("\tTraining Logistic Regression using Conjugate Gradient Descent...");
			LogRegressionSGD conjugate = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.CONJUGATE);				
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				conjugate.train(trainSet.xs, trainSet.ys);			
				conjugateErrors.get(j).add(LogRegressionSGD.getError(conjugate, testSet.xs, testSet.ys));
			}
		
			System.out.println("\tTraining Logistic Regression using Newton's Method Gradient Descent...");
			LogRegressionSGD newton = new LogRegressionSGD(WisconsinBreastCancerData.NUM_FIELDS, Variant.CONJUGATE);				
			for (int j = 0; j < MAX_ITERATIONS; j++) {
				newton.train(trainSet.xs, trainSet.ys);			
				newtonErrors.get(j).add(LogRegressionSGD.getError(newton, testSet.xs, testSet.ys));
			}		
		}

		DataTable batchData = new DataTable(Integer.class, Double.class);
		DataTable stochasticData = new DataTable(Integer.class, Double.class);
		DataTable momentumData = new DataTable(Integer.class, Double.class);
		DataTable conjugateData = new DataTable(Integer.class, Double.class);
		DataTable newtonData = new DataTable(Integer.class, Double.class);
		for (int i = 0; i < MAX_ITERATIONS; i++) {
			batchData.add(i, batchErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble());
			stochasticData.add(i, stochasticErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble());
			momentumData.add(i, momentumErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble());
			conjugateData.add(i, conjugateErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble());
			newtonData.add(i, newtonErrors.get(i).stream().mapToDouble(d -> d).average().getAsDouble());
		}

		DataSeries batchDS = new DataSeries("Batch", batchData, 0, 1);
		DataSeries stochasticDS = new DataSeries("Stochastic", stochasticData, 0, 1);
		DataSeries momentumDS = new DataSeries("Momentum", momentumData, 0, 1);
		DataSeries conjugateDS = new DataSeries("Conjugate", conjugateData, 0, 1);
		DataSeries newtonDS = new DataSeries("Conjugate", conjugateData, 0, 1);
		XYPlot plot = new XYPlot(batchDS, stochasticDS, momentumDS /*, conjugateDS, newtonDS*/);
		
		setLineRenderer(plot, batchDS, PlotUtils.BLUE);
		setLineRenderer(plot, stochasticDS, PlotUtils.GOLD);
		setLineRenderer(plot, momentumDS, PlotUtils.RED);
		setLineRenderer(plot, conjugateDS, PlotUtils.GREEN);
		setLineRenderer(plot, newtonDS, Color.ORANGE);
		
		plot.setInsets(new Insets2D.Double(20.0, 40.0, 40.0, 40.0));
		plot.getTitle().setText("Gradient Descent Algorithms: Error vs # Iterations");	
		plot.setLegendVisible(true);
	
		PlotUtils.drawPlot(plot);

	}

	private static void setLineRenderer(XYPlot plot, DataSeries ds, Color color) {
		LineRenderer lr = new DefaultLineRenderer2D();
		lr.setColor(color);
		plot.setLineRenderers(ds, lr);
		plot.setPointRenderers(ds, null);
	}
}
