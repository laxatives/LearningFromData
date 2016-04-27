package com.diffbot.learningfromdata.regression;

import java.io.IOException;
import java.util.Arrays;

import com.diffbot.learningfromdata.data.Data.Labelset;
import com.diffbot.learningfromdata.data.RedWineQualityData;
import com.diffbot.learningfromdata.data.WhiteWineQualityData;
import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.MathUtils.QRDecomp;
import com.diffbot.learningfromdata.utils.Utils;

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
		double[][] x_padded = RegressionModel.padBias(x);
		
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

	@Override
	public double eval(double[] x) {
		return MathUtils.dotProduct(w, RegressionModel.padBias(x));
	}
	
	@Override
	public double error(double h, double y) {
		return y - h;
	}

	private static final boolean WHITE_WINE = true; // else red wine
	private static final boolean DEBUG = false;
	
	public static void main(String[] args) throws IOException {
		Labelset es = WHITE_WINE ? (new WhiteWineQualityData()).getLabelset() : 
				(new RedWineQualityData()).getLabelset();
		
		System.out.println("Running Ordinary Least Squares on " + (WHITE_WINE ? "White" : "Red")
				+ " Wine Quality Dataset...");		
			
		OLSRegression model = new OLSRegression(es.xs, es.ys);
		double[] stats = RegressionModel.getStats(model, es.xs, es.ys);
		System.out.println("\t{MSE, Bias, Var}: " + Utils.arrayToString(stats));
		
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
