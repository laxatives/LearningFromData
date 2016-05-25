package com.diffbot.learningfromdata.net;

public enum ActivationFunction {
	TANH {
		@Override
		public double eval(double x) {
			return Math.tanh(x);
		}
		@Override
		public double derivative(double x) {
			return 1 - Math.pow(Math.tanh(x), 2);
		}
	},
	// from LeCun, Generalization and network design strategies
	// has variance = 1 for transformed inputs, gain ~= 1
	SCALED_TANH {
		@Override
		public double eval(double x) {
			return 1.7159 * Math.tanh(2 * x / 3.0);
		}
		@Override
		public double derivative(double x) {
			return 1.14393 * (1 - Math.pow(Math.tanh(2 * x / 3.0), 2)); 
		}			
	},
	// from LeCun, Efficient Back-prop
	TWISTED_TANH {
		private static final double a = 0.01;
		@Override
		public double eval(double x) {
			return Math.tanh(x) + a * x;
		}
		@Override
		public double derivative(double x) {
			return 1 - Math.pow(Math.tanh(x), 2) + a;
		}
	},
	RELU {
		@Override
		public double eval(double x) {
			return Math.max(0, x);
		}

		@Override
		public double derivative(double x) {
			return x < 0 ? 0 : 1;
		}
	},
	LEAKY_RELU {
		private static final double a = 0.01;
		@Override
		public double eval(double x) {
			return x < 0 ? a * x : x; 
		}
		@Override
		public double derivative(double x) {
			return x < 0 ? a : 1;
		}
	};
	
	public abstract double eval(double x);
	public abstract double derivative(double x);
}