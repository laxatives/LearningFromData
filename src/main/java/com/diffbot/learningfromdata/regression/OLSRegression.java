package com.diffbot.learningfromdata.regression;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import com.diffbot.learningfromdata.data.DataSet.TrainingExamples;
import com.diffbot.learningfromdata.data.RedWineQualityData;
import com.diffbot.learningfromdata.data.WhiteWineQualityData;
import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.PlotUtils;
import com.diffbot.learningfromdata.utils.MathUtils.QRDecomp;
import com.diffbot.learningfromdata.utils.Utils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;

/**
 * Applies Ordinary Least Squares to Wine Quality data sets (red or white).
 */
public class OLSRegression implements RegressionModel {
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
		
		if (Arrays.stream(w).anyMatch(d -> Double.isNaN(d))) {
			System.out.println("WARNING: weights array contains NaN");
		}
	}

	public double eval(double[] x) {
		return MathUtils.dotProduct(w, padBias(x));
	}
	
	public static double[] padBias(double[] x) {
		double[] p = new double[x.length + 1];
		p[0] = 1;
		System.arraycopy(x, 0, p, 1, x.length);
		return p;
	}
	
	public static double[][] padBias(double[][] x) {
		double[][] p = new double[x.length][x[0].length + 1];
		for (int i = 0; i < x.length; i++) {
			double[] padded = padBias(x[i]);
			p[i] = padded;
		}
		return p;
	}
	
	private static final boolean GENERATED_DATA = false; // else red/white wine
	private static final boolean WHITE_WINE = true; // else red wine
	private static final int NUM_FIELDS = 11;
	private static Random RANDOM = new Random();	
	private static final boolean DEBUG = false;
	
	private static TrainingExamples getGeneratedExamples(int numExamples, int numFields, double std) {
		double[] trueWeights = new double[numFields + 1];
		for (int i = 0; i < numFields + 1; i++) {
			trueWeights[i] = RANDOM.nextDouble();
		}
		
		double[][] xs = new double[numExamples][numFields];
		double[] ys = new double[numExamples];
		
		for (int i = 0; i < numExamples; i++) {
			for (int j = 0; j < numFields; j++) {
				xs[i][j] = RANDOM.nextDouble();
			}
			ys[i] = MathUtils.dotProduct(padBias(xs[i]), trueWeights) + std * RANDOM.nextGaussian();
		}
		return new TrainingExamples(xs, ys);
	}
	
	public double[] printStats(double[][] x, double[] y_t) {		
		double[] estimates = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			estimates[i] = eval(x[i]);
		}		
		double avg_estimate = Arrays.stream(estimates).sum() / x.length;
		
		double mse = 0;
		double var = 0;
		double bias = 0;
		for (int i = 0; i < x.length; i++) {
			mse += Math.pow(estimates[i] - y_t[i], 2); 
			var += Math.pow(estimates[i]- avg_estimate, 2);
			bias += Math.pow(avg_estimate - y_t[i], 2);
		}
		mse = mse / x.length;
		var = var / x.length;
		bias = bias / x.length;
		
		System.out.println("\tvar: " + var);
		System.out.println("\tbias: " + bias);
		System.out.println("\tv+b: " + (var + bias));
		System.out.println("\tmse: " + mse);
		System.out.println("\tire: " + Math.abs(mse - (var + bias)));		
		
		return new double[]{var, bias, mse};
	}
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException {
		DataTable var = new DataTable(Integer.class, Double.class);
		DataTable bias = new DataTable(Integer.class, Double.class);
		DataTable vb = new DataTable(Integer.class, Double.class);
		DataTable mse = new DataTable(Integer.class, Double.class);
		for (int i = 0; i <= NUM_FIELDS; i++) {
			System.out.println(String.format("Training OLSRegression using using wine-quality dataset on " + i + " fields..."));
			TrainingExamples es = GENERATED_DATA ? getGeneratedExamples(3, i , 0) :
					WHITE_WINE ? (new WhiteWineQualityData()).getTrainingExamples() : 
						(new RedWineQualityData()).getTrainingExamples();
			
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
		plot.getTitle().setText("Ordinary Least Squares: Bias, Variance, MSE");	
		plot.setLegendVisible(true);
	
//		PlotUtils.drawPlot(plot);
		
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
