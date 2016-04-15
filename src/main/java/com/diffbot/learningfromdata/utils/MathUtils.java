package com.diffbot.learningfromdata.utils;

public class MathUtils {
	public static double[] normalize(double[] xs) {
		double l2norm = Math.sqrt(dotProduct(xs, xs));
		return scalarProduct(1.0 / l2norm, xs);
	}
	
	public static double[] scalarProduct(double x, double[] xs) {
		double[] product = new double[xs.length];
		for (int i = 0; i < product.length; i++) {
			product[i] = x * xs[i];
		}
		return product;
	}	
			
	public static double dotProduct(double[] xs1, double[] xs2) {
		double product = 0;		
		for (int i = 0; i < xs1.length; i++) {
			product += xs1[i] * xs2[i];
		}
		return product;
	}
	
	/**
	 * Returns the projection of v in the u direction.
	 */
	public static double[] project(double[] u, double[] v) {
		return scalarProduct(dotProduct(u, v) / dotProduct(u, u), u);
	}
	
	public static double[] sumArrays(double[] xs1, double[] xs2) {
		double[] sum = new double[xs1.length];
		for (int i = 0; i < xs1.length; i++) {
			sum[i] = xs1[i] + xs2[i];
		}
		return sum;
	}
	
	public static double[][] transpose(double[][] x) {
		double[][] t = new double[x[0].length][x.length];
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[0].length; j++) {
				t[j][i] = x[i][j];
			}
		}
		return t;
	}
	
	public static double[][] multiply(double[][] a, double[][] x) {
		double[][] y = new double[a.length][x[0].length];
		double[][] x_t = transpose(x);
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < x_t.length; j++) {
				y[i][j] = dotProduct(a[i], x_t[j]);	
			}
		}
		return y;
	}
	
	public static class QRDecomp {
		public final double[][] Q;
		public final double[][] R;
		public QRDecomp(double[][] Q, double[][] R) {
			this.Q = Q;
			this.R = R;
		}
	}
	
	/**
	 * Returns the R (upper triangular) component of a QR Decomposition on the input.
	 */
	public static QRDecomp qrDecompose(double[][] a) {
		double[][] a_t = transpose(a);
		
		// get orthonormal basis (same as Q transpose)
		double[][] e = modifiedGramSchmidt(a_t);
		
		// transform a to new basis
		double[][] a_rot = new double[a[0].length][a.length];  
		for (int i = 0; i < a_rot.length; i++) {
			for (int j = 0; j <= i; j++) {
				a_rot[i] = sumArrays(a_rot[i], 
						MathUtils.scalarProduct(MathUtils.dotProduct(e[j], a_t[i]), e[j]));
			}
		}
		
		// generate r
		double[][] r = new double[a.length][a[0].length];
		for (int i = 0; i < r.length; i++) {
			for (int j = i; j < r[0].length; j++) {
				r[i][j] = MathUtils.dotProduct(e[i], a_t[j]);
			}
		}
		
		return new QRDecomp(MathUtils.transpose(e), r);
	}
	
	/**
	 * Orthonormalizes a set of vectors.
	 */
	public static double[][] modifiedGramSchmidt(double[][] s) {
		// TODO: use Householder reflections for numerical stability
		double[][] u = new double[s.length][s[0].length];
		// orthogonalize
		for (int i = 0; i < s.length; i++) {
			u[i] = s[i];
			for (int j = 0; j < i; j++) {
				u[i] = sumArrays(u[i], scalarProduct(-1, project(u[j], u[i])));
			}
		}
		// normalize
		for (int i = 0; i < u.length; i++) {
			u[i] = normalize(u[i]);
		}
		return u;
	}
	
	/**
	 * Uses backward substitution to invert an upper triangular matrix.
	 */
	public static double[][] invertUpperTriangular(double[][] r) {
		int dim = Math.min(r.length, r[0].length);
		double[][] inv = new double[dim][dim];
		
		for (int i = dim - 1; i >= 0; i--) {
			inv[i][i] = 1.0 / r[i][i];
			for (int j = i + 1; j < dim; j++) {
				for (int k = i + 1; k < dim; k++) {
					inv[i][j] -= r[i][k] * inv[k][j];
				}
				inv[i][j] *=  inv[i][i];
			}			
		}
		return inv;
	}
}
