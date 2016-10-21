package com.diffbot.learningfromdata.net;

import java.util.List;

public enum LossFunction {
    SVM {
        @Override
        float classificationError(List<Float> results, int correctIndex) {
            if (correctIndex >= results.size()) {
                throw new IllegalArgumentException("Cannot have correctIndex " + correctIndex + " for results " + results);
            }
            float err = 0;
            for (int i = 0; i < results.size(); i++) {
                if (i == correctIndex) {
                    continue;
                }
                err += Math.max(0, results.get(i) - results.get(correctIndex) + 1);
            }

            return err;
        }
    },
    SOFT_MAX {
        @Override
        float classificationError(List<Float> results, int correctIndex) {
            if (correctIndex >= results.size()) {
                throw new IllegalArgumentException("Cannot have correctIndex " + correctIndex + " for results " + results);
            }
            float err = 0;
            for (int i = 0; i < results.size(); i++) {
                if (i == correctIndex) {
                    continue;
                }
                err += Math.max(0, results.get(i) - results.get(correctIndex) + 1);
            }

            return err;
        }
    },
    ;
    abstract float classificationError(List<Float> results, int correctIndex);
}
