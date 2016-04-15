package com.diffbot.learningfromdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.PlotUtils;
import com.diffbot.learningfromdata.utils.MathUtils.QRDecomp;
import com.diffbot.learningfromdata.utils.Utils;
import com.diffbot.learningfromdata.utils.Utils.TrainingExamples;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;

public class OLSRegression {

	private static final String DATA_DIR = "/home/bhan/workspace/LearningFromData/src/main/resources";
	private static final File DATA_FILE_WHITE = new File(DATA_DIR, "winequality-white.csv");
	private static final int NUM_EXAMPLES_WHITE = 4898;
	private static final File DATA_FILE_RED = new File(DATA_DIR, "winequality-red.csv");
	private static final int NUM_EXAMPLES_RED = 1599;
	private static final int NUM_FIELDS = 11; // 11
	private static Random RANDOM = new Random();
	
	private static final boolean DEBUG = false;
	
	public double[] w;
	
	/**
	 * Ordinary Least Squares using QR decomposition.
	 * See https://inst.eecs.berkeley.edu/~ee127a/book/login/l_ols_ls_def.html
	 */
	public OLSRegression(double[][] x, double[] y_t) {
		double[][] x_padded = padBias(x);
		
		QRDecomp qr = MathUtils.qrDecompose(x_padded);
		double[][] q_t = MathUtils.transpose(qr.Q);
		double[][] r_i = MathUtils.invertUpperTriangular(qr.R);
		
		double[][] inv = MathUtils.multiply(r_i, q_t);
		double[][] y = MathUtils.transpose(new double[][] {y_t});
		w = MathUtils.transpose(MathUtils.multiply(inv, y))[0];
	}

	public double eval(double[] x) {
		return MathUtils.dotProduct(w, padBias(x));
	}
	
	private static double[] padBias(double[] x) {
		double[] p = new double[x.length + 1];
		p[0] = 1;
		System.arraycopy(x, 0, p, 1, x.length);
		return p;
	}
	
	private static double[][] padBias(double[][] x) {
		double[][] p = new double[x.length][x[0].length + 1];
		for (int i = 0; i < x.length; i++) {
			double[] padded = padBias(x[i]);
			p[i] = padded;
		}
		return p;
	}

	private static TrainingExamples getTrainingExamples(File dataFile, int numExamples, int numFields) throws IOException {
		double[][] xs = new double[numExamples][numFields];
		double[] ys = new double[numExamples];
		try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
			int i = 0;
			String line = br.readLine(); // skip header
		    while ((line = br.readLine()) != null) {
		    	String[] features = line.split(";");
		    	double[] x = Arrays.stream(features).limit(numFields).map(Double::valueOf).mapToDouble(Double::doubleValue).toArray();
		    	Double y = Double.valueOf(features[numFields]);
		    	
		    	xs[i] = x;
		    	ys[i] = y;
		    	i++;
		    }
		}
		return new TrainingExamples(xs, ys);
	}
	
	private static TrainingExamples getGeneratedExamples(int numExamples, int numFields, double std) {
		double[] trueWeights = new double[numFields];
		for (int i = 0; i < numFields; i++) {
			trueWeights[i] = RANDOM.nextDouble();
		}
		
		double[][] xs = new double[numExamples][numFields];
		double[] ys = new double[numExamples];
		
		for (int i = 0; i < numExamples; i++) {
			for (int j = 0; j < numFields; j++) {
				xs[i][j] = RANDOM.nextDouble();
			}
			ys[i] = MathUtils.dotProduct(xs[i], trueWeights) + std * RANDOM.nextGaussian();
		}
		return new TrainingExamples(xs, ys);
	}
	
	public double[] printStats(double[][] x, double[] y_t) {		
		double[] guesses = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			guesses[i] = eval(x[i]);
		}		
		double avg_guess = Arrays.stream(guesses).sum() / x.length;
		double avg_y = Arrays.stream(y_t).sum() / y_t.length;
		
		double mse = 0;
		double var = 0;
		double bias = 0;
		for (int i = 0; i < x.length; i++) {
			mse += Math.pow(guesses[i] - y_t[i], 2); 
			var += Math.pow(guesses[i]- avg_guess, 2);
			bias += avg_guess - y_t[i];
		}
		mse = mse / x.length;
		var = var / x.length;
		
		System.out.println("\tvar: " + var);
		System.out.println("\tbias: " + bias);
		System.out.println("\tv+b: " + (var + Math.pow(bias, 2)));
		System.out.println("\tmse: " + mse);
		System.out.println("\tire: " + Math.abs(mse - var + Math.pow(bias, 2)));		
		
		return new double[]{var, bias, mse};
	}
	
	public static void main(String[] args) throws IOException {
		DataTable var = new DataTable(Integer.class, Double.class);
		DataTable bias = new DataTable(Integer.class, Double.class);
		DataTable vb = new DataTable(Integer.class, Double.class);
		DataTable mse = new DataTable(Integer.class, Double.class);
		for (int i = 1; i <= NUM_FIELDS; i++) {
			System.out.println(String.format("Training OLSRegression using using wine-quality dataset on " + i + " fields..."));
			TrainingExamples es = getGeneratedExamples(100_000, i, 0.2);
//			TrainingExamples es = getTrainingExamples(DATA_FILE_RED, NUM_EXAMPLES_RED, i);
//			TrainingExamples es = getTrainingExamples(DATA_FILE_WHITE, NUM_EXAMPLES_WHITE, i);
			
			OLSRegression model = new OLSRegression(es.xs, es.ys);
			double[] stats = model.printStats(es.xs, es.ys);
			var.add(i, stats[0]);
			bias.add(i, stats[1]);
			vb.add(i, stats[0] + stats[1]);
			mse.add(i, stats[2]);
		}
		
		DataSeries tpD2s = new DataSeries("Variance", var, 0, 1);
		DataSeries fpD2s = new DataSeries("Bias", bias, 0, 1);
		DataSeries tnD2s = new DataSeries("Var + Bias", vb, 0, 1);
		DataSeries fnD2s = new DataSeries("MSE", mse, 0, 1);
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
		
		if (DEBUG) {
			double[][] x = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
			QRDecomp qr = MathUtils.qrDecompose(x);
			System.out.println("X: " + Utils.matrixToString(x));
			System.out.println("QR: " + Utils.matrixToString(MathUtils.multiply(qr.Q, qr.R)));
			
			System.out.println("Q: " + Utils.matrixToString(qr.Q));
			System.out.println("Q^T * Q: " + Utils.matrixToString(MathUtils.multiply(MathUtils.transpose(qr.Q), qr.Q)));
			System.out.println("R: " + Utils.matrixToString(qr.R));
			System.out.println("R^-1: " + Utils.matrixToString(MathUtils.invertUpperTriangular(qr.R)));
		}
	}	
	
}
