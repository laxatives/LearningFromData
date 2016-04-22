package com.diffbot.learningfromdata;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.PlotUtils;
import com.diffbot.learningfromdata.utils.Utils;

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;

public class OLSBiasVariance {

	private static final int MAX_DOF = 16;
	private static final int NUM_EXAMPLES = 100;
	private static final double NOISE_STD = 0;
	private static Random RANDOM = new Random();
	
	private static double[] getStats(OLSRegression model, double[][] x, double[] y_t) {		
		double[] estimates = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			estimates[i] = model.eval(x[i]);
		}		
		double avg_estimate = Arrays.stream(estimates).sum() / x.length;
		
		double mse = 0;
		double var = 0;
		double bias = 0;
		for (int i = 0; i < x.length; i++) {
			mse += Math.pow(estimates[i] - y_t[i], 2); 
			var += Math.pow(estimates[i]- avg_estimate, 2);
			bias += avg_estimate - y_t[i];			
		}		
		
		mse /= x.length;
		var /= x.length;
		bias = Math.pow(bias, 2) / x.length;
		
		return new double[]{var, bias, mse};
	}
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException {		
		double[] trueWeights = new double[MAX_DOF + 1];
		double trueDOF = RANDOM.nextInt(MAX_DOF / 4) + (2 * MAX_DOF / 3);
		for (int i = 0; i < trueDOF; i++) {
			trueWeights[i] = RANDOM.nextDouble() + 1;
		}
		System.out.println("True DOF: " + trueDOF);
		System.out.println("True weights: " + Utils.arrayToString(trueWeights));
		
		double[][] completeXs = new double[NUM_EXAMPLES][MAX_DOF - 1];
		double[] ys = new double[NUM_EXAMPLES];
		double[][] holdoutXs = new double[NUM_EXAMPLES][MAX_DOF - 1];
		double[] holdoutYs = new double[NUM_EXAMPLES];
		
		// compute X to evaluate true Y for each sample
		for (int i = 0; i < NUM_EXAMPLES; i++) {
			double x = 0.5 * RANDOM.nextGaussian() + 1;
			completeXs[i][0] = x;
			for (int j = 1; j < completeXs[i].length; j++) {
				completeXs[i][j] = completeXs[i][0] * completeXs[i][j - 1];
			}			
			ys[i] = MathUtils.dotProduct(OLSRegression.padBias(completeXs[i]), trueWeights) + NOISE_STD * RANDOM.nextGaussian();

			double hx = 0.5 * RANDOM.nextGaussian() + 1;
			holdoutXs[i][0] = hx;
			for (int j = 1; j < holdoutXs[i].length; j++) {
				holdoutXs[i][j] = holdoutXs[i][0] * holdoutXs[i][j - 1];
			}			
			holdoutYs[i] = MathUtils.dotProduct(OLSRegression.padBias(holdoutXs[i]), trueWeights) + NOISE_STD * RANDOM.nextGaussian();
		}		
		
		DataTable var = new DataTable(Integer.class, Double.class);
		DataTable bias = new DataTable(Integer.class, Double.class);
		DataTable vb = new DataTable(Integer.class, Double.class);
		DataTable mse = new DataTable(Integer.class, Double.class);
		for (int dof = 0; dof < MAX_DOF; dof++) {
			System.out.println(String.format("Training OLSRegression using generated dataset with " + dof + " DOF..."));
			
			// need to resize xs at every iteration since OLS input must be full-rank
			double[][] xs = new double[NUM_EXAMPLES][dof];
			double[][] testXs = new double[NUM_EXAMPLES][dof];
			for (int i = 0; i < xs.length; i++) {
				for (int j = 0; j < xs[0].length; j++) {
					xs[i][j] = completeXs[i][j];
					testXs[i][j] = holdoutXs[i][j];
				}
			}
			
			OLSRegression model = new OLSRegression(xs, ys);
			double[] stats = getStats(model, holdoutXs, holdoutYs);
			var.add(dof, stats[0]);
			bias.add(dof, stats[1]);
			vb.add(dof, stats[0] + stats[1]);
			mse.add(dof, stats[2]);
			System.out.println(Utils.arrayToString(stats));
		}
		
		System.out.println("True DOF: " + trueDOF);
		System.out.println("True weights: " + Utils.arrayToString(trueWeights));
		
		DataSeries varDS = new DataSeries("Variance", var, 0, 1);
		DataSeries biasDS = new DataSeries("Bias", bias, 0, 1);
		DataSeries vbDS = new DataSeries("Var + Bias", vb, 0, 1);
		DataSeries mseDS = new DataSeries("MSE", mse, 0, 1);
		XYPlot plot = new XYPlot(varDS, biasDS, vbDS, mseDS);
		
		LineRenderer varR = new DefaultLineRenderer2D();
		varR.setColor(PlotUtils.BLUE);
		plot.setLineRenderers(varDS, varR);
		LineRenderer biasR = new DefaultLineRenderer2D();
		biasR.setColor(PlotUtils.GOLD);
		plot.setLineRenderers(vbDS, biasR);		
		LineRenderer vbR = new DefaultLineRenderer2D();
		vbR.setColor(PlotUtils.GREEN);
		plot.setLineRenderers(biasDS, vbR);
		LineRenderer mseR = new DefaultLineRenderer2D();
		mseR.setColor(PlotUtils.RED);
		plot.setLineRenderers(mseDS, mseR);

		plot.setInsets(new Insets2D.Double(20.0, 40.0, 40.0, 40.0));
		plot.getTitle().setText("Ordinary Least Squares: Bias, Variance, MSE");	
		plot.setLegendVisible(true);
	
		PlotUtils.drawPlot(plot);
	}	
}
