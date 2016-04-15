package com.diffbot.learningfromdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import com.diffbot.learningfromdata.utils.MathUtils;
import com.diffbot.learningfromdata.utils.MathUtils.QRDecomp;
import com.diffbot.learningfromdata.utils.Utils;
import com.diffbot.learningfromdata.utils.Utils.TrainingExamples;

public class OLSRegression {

	private static final String DATA_DIR = "/home/bhan/workspace/LearningFromData/src/main/resources";
	private static final File DATA_FILE_WHITE = new File(DATA_DIR, "winequality-white.csv");
	private static final int NUM_EXAMPLES_WHITE = 4898;
	private static final File DATA_FILE_RED = new File(DATA_DIR, "winequality-red.csv");
	private static final int NUM_EXAMPLES_RED = 1599;
	private static final int NUM_FIELDS = 11; // 11
	private static Random RANDOM = new Random();
	
	private static final boolean DEBUG = false;
	
	double[] w;
	
	/**
	 * Ordinary Least Squares using QR decomposition.
	 * See https://inst.eecs.berkeley.edu/~ee127a/book/login/l_ols_ls_def.html
	 */
	public OLSRegression(double[][] x, double[] y_t, int numParameters) {
		// TODO: use numParameters
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

	private static TrainingExamples getTrainingExamples(File dataFile) throws IOException {
		double[][] xs = new double[NUM_EXAMPLES_WHITE][NUM_FIELDS];
		double[] ys = new double[NUM_EXAMPLES_WHITE];
		try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
			int i = 0;
			String line = br.readLine(); // skip header
		    while ((line = br.readLine()) != null) {
		    	String[] features = line.split(";");
		    	double[] x = Arrays.stream(features).limit(NUM_FIELDS).map(Double::valueOf).mapToDouble(Double::doubleValue).toArray();
		    	Double y = Double.valueOf(features[NUM_FIELDS]);
		    	
		    	xs[i] = x;
		    	ys[i] = y;
		    	i++;
		    }
		}
		return new TrainingExamples(xs, ys);
	}
	
	private static TrainingExamples getGeneratedExamples() {
		double[] trueWeights = new double[NUM_FIELDS];
		for (int i = 0; i < NUM_FIELDS; i++) {
			trueWeights[i] = RANDOM.nextDouble();
		}
		
		double[][] xs = new double[NUM_EXAMPLES_WHITE][NUM_FIELDS];
		double[] ys = new double[NUM_EXAMPLES_WHITE];
		
		for (int i = 0; i < NUM_EXAMPLES_WHITE; i++) {
			for (int j = 0; j < NUM_FIELDS; j++) {
				xs[i][j] = RANDOM.nextDouble();
			}
			ys[i] = MathUtils.dotProduct(xs[i], trueWeights);
		}
		return new TrainingExamples(xs, ys);
	}
	
	public void printStats(double[][] x, double[] y_t) {		
		double[] guesses = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			guesses[i] = eval(x[i]);
		}		
		double avg_guess = Arrays.stream(guesses).sum() / x.length;
		
		double mse = 0;
		double var = 0;
		double bias = 0;
		for (int i = 0; i < x.length; i++) {
			double se = Math.pow(guesses[i] - y_t[i], 2); 
			mse += se;
			var += Math.pow(guesses[i]- avg_guess, 2);
			bias += Math.pow(avg_guess - y_t[i], 2);
		}
		mse = mse / x.length;
		var = var / x.length;
		bias = bias / x.length;
		
		System.out.println("var: " + var);
		System.out.println("bias: " + bias);
		System.out.println("v+b: " + (var + bias));
		System.out.println("mse: " + mse);
	}
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException {
		System.out.println(String.format("Training OLSRegression using using wine-quality dataset..."));
		TrainingExamples es = getTrainingExamples(DATA_FILE_WHITE);
//		TrainingExamples es = getGeneratedExamples();
		OLSRegression model = new OLSRegression(es.xs, es.ys, NUM_FIELDS);
		
		model.printStats(es.xs, es.ys);
		
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
