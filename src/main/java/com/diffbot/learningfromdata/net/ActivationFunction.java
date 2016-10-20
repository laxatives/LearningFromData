package com.diffbot.learningfromdata.net;

public enum ActivationFunction {
	TANH {
		@Override
		public float eval(float x) {
			return (float) Math.tanh(x);
		}
		@Override
		public float derivative(float x) {
			return 1 - (float) Math.pow(Math.tanh(x), 2);
		}
	},
	// from LeCun, Generalization and network design strategies
	// has variance = 1 for transformed inputs, gain ~= 1
	SCALED_TANH {
		@Override
		public float eval(float x) {
			return (float) (1.7159 * Math.tanh(2 * x / 3.0));
		}
		@Override
		public float derivative(float x) {
			return (float) (1.14393f * (1 - Math.pow(Math.tanh(2 * x / 3.0), 2)));
		}			
	},
	// from LeCun, Efficient Back-prop
	TWISTED_TANH {
		private static final float a = 0.01f;
		@Override
		public float eval(float x) {
			return (float) Math.tanh(x) + a * x;
		}
		@Override
		public float derivative(float x) {
			return 1 - (float) Math.pow(Math.tanh(x), 2) + a;
		}
	},
	RELU {
		@Override
		public float eval(float x) {
			return Math.max(0, x);
		}

		@Override
		public float derivative(float x) {
			return x < 0 ? 0 : 1;
		}
	},
	LEAKY_RELU {
		private static final float a = 0.01f;
		@Override
		public float eval(float x) {
			return x < 0 ? a * x : x; 
		}
		@Override
		public float derivative(float x) {
			return x < 0 ? a : 1;
		}
	};
    
    // TODO: add maxout, elu
	
	public abstract float eval(float x);
	public abstract float derivative(float x);
}