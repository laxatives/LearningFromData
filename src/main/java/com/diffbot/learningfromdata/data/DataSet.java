package com.diffbot.learningfromdata.data;

import java.io.IOException;

public interface DataSet {
	
	TrainingExamples getTrainingExamples() throws IOException;

	public static class TrainingExamples {
		public double[][] xs;
		public double[] ys;
		public TrainingExamples(double[][] xs, double[] ys) {
			this.xs = xs;
			this.ys = ys;
		}
	}
}
