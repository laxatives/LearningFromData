package com.diffbot.ml;

import com.diffbot.thoth.utilities.Triple;
import com.diffbot.toolbox.FileTools;
import com.diffbot.utils.Pair;
import com.esotericsoftware.minlog.Log;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableSet;
import com.koloboke.collect.map.hash.*;
import com.koloboke.collect.set.hash.HashIntSet;
import com.koloboke.collect.set.hash.HashIntSets;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Combines multi-hop paths using vector addition. This appears to be the best scoring operator in
 * {ADD, MUL, RNN}.
 *
 * TODO: refactor this and make it sane
 * TODO: port to Spark/Tensorflow
 */
public class PTransEAddTrain {
    private static final Splitter WHITESPACE_SPLITTER = Splitter.onPattern("\\s+").trimResults();
    private static final float RATE = 0.001f;
    private static final float MARGIN = 1f;

    static final boolean L1_FLAG = true;
    static final int N = 100;
    static final File ENTITY2VEC_FILE = new File(KGCompletion.KB2E_DIRECTORY, "entity2vec.txt");
    static final File RELATION2VEC_FILE = new File(KGCompletion.KB2E_DIRECTORY, "relation2vec.txt");

    // TODO: use koloboke maps/lists
    private int entityCount = 0;
    private int relationCount = 0;
    private HashObjIntMap<String> entityToId = HashObjIntMaps.newMutableMap();
    private HashObjFloatMap<Pair<String, Integer>> pathConfidence = HashObjFloatMaps.newMutableMap();
    private List<Integer> headIds = new ArrayList<>(); // list of headIds
    private List<Integer> tailIds = new ArrayList<>(); // list of tailIds
    private List<Integer> relationIds = new ArrayList<>(); // list of relationIds
    private List<List<Pair<int[], Float>>> fb_path = new ArrayList<>();

    /**
     * A map of Pair<headId, relationId> -> Set<tailId> where every headId, readId, tailId triple
     * implies a positive label.
     */
    private HashObjObjMap<Pair<Integer, Integer>, HashIntSet> positiveTriples =
            HashObjObjMaps.newMutableMap();

    private float error;
    private float[][] entityTmp;
    private float[][] entityVec;
    private float[][] relationTmp;
    private float[][] relationVec;


    public void train() throws IOException {
        prepare();
        run();
    }

    private void run() throws IOException {
        Log.info("PTransEAddTrain.run", "n=" + N + " RATE=" + RATE);

        entityTmp = new float[entityCount][N];
        entityVec = new float[entityCount][N];

        double range = 6.0 / Math.sqrt(N);
        for (int i = 0; i < entityCount; i++) {
            for (int j = 0; j < N; j++) {
                entityVec[i][j] = (float) randn(0, 1.0 / N, -1 * range, range);
            }
            norm(entityVec[i]);
        }

        relationTmp = new float[relationCount][N];
        relationVec = new float[relationCount][N];
        for (int i = 0; i< relationCount; i++) {
            for (int j = 0; j < N; j++) {
                relationVec[i][j] = (float) randn(0, 1.0 / N, -1 * range, range);
            }
        }

        bfgs();
    }

    /**
     * Quasi-Newton method for gradient-descent (short for Broyden–Fletcher–Goldfarb–Shanno)
     */
    private void bfgs() throws IOException {
        long startMs = System.currentTimeMillis();
        Log.info("PTransEAddTrain.bfgs", "margin=" + MARGIN);
        int nbatches = 100;
        int neval = 1000;
        int batch_size = headIds.size()/nbatches;

        relationTmp = deepCopy(relationVec);
        entityTmp = deepCopy(entityVec);

        for (int eval = 0; eval < neval; eval++) {
            error = 0;
            for (int batch = 0; batch < nbatches; batch++) {
                IntStream.range(0, batch_size).parallel().forEach(k -> {
                    int i = randMax(headIds.size());
                    int headId = headIds.get(i);
                    int tailId = tailIds.get(i);
                    int relationId = relationIds.get(i);

                    // For every positive label, we generate a negative label through corruption
                    Triple<Integer, Integer, Integer> corruptTriple =
                            corruptPositiveTriple(headId, relationId, tailId);
                    int corruptHeadId = corruptTriple.first;
                    int corruptRelationId = corruptTriple.second;
                    int corruptTailId = corruptTriple.third;

                    trainKb(headId, relationId, tailId, corruptHeadId, corruptRelationId, corruptTailId);

                    if (!fb_path.get(i).isEmpty()) {
                        int rel_neg = randMax(relationCount);
                        while (positiveTripleExists(new Pair<>(headId, rel_neg), tailId)) {
                            rel_neg = randMax(relationCount);
                        }

                        for (int path_id = 0; path_id < fb_path.get(i).size(); path_id++) {
                            int[] rel_path = fb_path.get(i).get(path_id).first;
                            String path = Arrays.stream(rel_path)
                                    .boxed()
                                    .map(String::valueOf)
                                    .collect(Collectors.joining(" "));
                            double pr = fb_path.get(i).get(path_id).second;
                            double pr_path = 0;
                            if (pathConfidence.containsKey(new Pair<>(path, relationId))) {
                                pr_path = pathConfidence.getFloat(new Pair<>(path, relationId));
                            }
                            pr_path = 0.99 * pr_path + 0.01;
                            train_path(relationId, rel_neg, rel_path, 2 * MARGIN, pr * pr_path);
                        }
                    }

                    // Rescale any modified vectors
                    for (int entityId : ImmutableSet.of(headId, tailId, corruptHeadId, corruptTailId)) {
                        norm(entityTmp[entityId]);
                    }
                    for (int rId : ImmutableSet.of(relationId, corruptRelationId)) {
                        norm(relationTmp[rId]);
                    }
                });
                relationVec = deepCopy(relationTmp);
                entityVec = deepCopy(entityTmp);
            }

            Log.info("PTransEAddTrain.bfgs", "eval " + eval + " of " + neval +
                    ", error=" + error + ", time=" + (System.currentTimeMillis() - startMs) + "ms");
            try (BufferedWriter relationVecWriter = FileTools.bufferedWriter(RELATION2VEC_FILE)) {
                for (int i = 0; i < relationCount; i++) {
                    for (int j = 0; j < N; j++) {
                        relationVecWriter.write(String.format("%.6f\t", relationVec[i][j]));
                    }
                    relationVecWriter.write("\n");
                }
            }

            try (BufferedWriter entityVecWriter = FileTools.bufferedWriter(ENTITY2VEC_FILE)) {
                for (int i = 0; i < entityCount; i++) {
                    for (int j = 0; j < N; j++) {
                        entityVecWriter.write(String.format("%.6f\t", entityVec[i][j]));
                    }
                    entityVecWriter.write("\n");
                }
            }

        }
    }

    /**
     * Corrupts the provided positive label by randomly changing one of {headId, relationId, tailId}
     * such that the new triple is not in the positive label set.
     */
    private Triple<Integer,Integer,Integer> corruptPositiveTriple(int headId, int relationId, int tailId) {
        double tmp = Math.random();
        if (tmp < 0.25) {
            // Corrupt the tail entity
            Pair<Integer, Integer> key = new Pair<>(headId, relationId);

            int corruptEntityId = randMax(entityCount);
            while (positiveTripleExists(key, corruptEntityId)) {
                corruptEntityId = randMax(entityCount);
            }
            return new Triple<>(headId, relationId, corruptEntityId);
        } else if (tmp < 0.5) {
            // Corrupt the head entity
            int corruptEntityId = randMax(entityCount);
            while (positiveTripleExists(new Pair<>(corruptEntityId, relationId), tailId)) {
                corruptEntityId = randMax(entityCount);
            }

            return new Triple<>(corruptEntityId, relationId, tailId);
        } else {
            // Corrupt the relation
            int corruptRelationId = randMax(relationCount);
            while (positiveTripleExists(new Pair<>(headId, corruptRelationId), tailId)) {
                corruptRelationId = randMax(relationCount);
            }
            return new Triple<>(headId, corruptRelationId, tailId);
        }
    }

    private boolean positiveTripleExists(Pair<Integer, Integer> key, int tailId) {
        return positiveTriples.containsKey(key) && positiveTriples.get(key).contains(tailId);
    }

    private void train_path(int rel, int rel_neg, int[] rel_path, double margin, double x) {
        double sum1 = calc_path(rel, rel_path);
        double sum2 = calc_path(rel_neg, rel_path);
        double lambda = 1;
        if (sum1 + margin > sum2) {
            error += x * lambda * (margin + sum1 - sum2);
            gradient_path(rel, rel_path, -x * lambda);
            gradient_path(rel_neg, rel_path, x * lambda);
        }
    }

    private double calc_path(int r1, int[] rel_path) {
        double sum = 0;
        for (int k = 0; k < N; k++) {
            double tmp = relationVec[r1][k];
            for (int j = 0; j < rel_path.length; j++) {
                tmp -= relationVec[rel_path[j]][k];
            }

            if (L1_FLAG) {
                sum += Math.abs(tmp);
            } else {
                sum += tmp * tmp;
            }
        }
        return sum;
    }

    private void gradient_path(int r1, int[] rel_path, double delta) {
        for (int k = 0; k < N; k++) {
            double x = relationVec[r1][k];
            for (int j = 0; j<rel_path.length; j++) {
                x -= relationVec[rel_path[j]][k];
            }

            if (L1_FLAG) {
                if (x > 0) {
                    x = 1;
                } else {
                    x = -1;
                }
            }

            synchronized (relationTmp[r1]) {
                relationTmp[r1][k] += delta * RATE * x;
            }
            for (int j = 0; j < rel_path.length; j++) {
                synchronized (relationTmp[rel_path[j]]) {
                    relationTmp[rel_path[j]][k] -= delta * RATE * x;
                }
            }
        }
    }

    /**
     * Compute a hinge loss on the true label and the corrupt label and perform gradient descent if
     * necessary.
     */
    private void trainKb(int trueHeadId, int trueRelationId, int trueTailId,
            int corruptHeadId, int corruptRelationId, int corruptTailId) {
        double trueScore = calc_kb(trueHeadId, trueTailId, trueRelationId);
        double corruptScore = calc_kb(corruptHeadId, corruptTailId, corruptRelationId);
        if (trueScore + MARGIN > corruptScore) {
            error += MARGIN + trueScore - corruptScore;
            gradient_kb(trueHeadId, trueRelationId, trueTailId, -1);
            gradient_kb(corruptHeadId, corruptRelationId, corruptTailId, 1);
        }
    }

    private double calc_kb(int e1,int e2,int rel) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            double tmp = entityVec[e2][j] - entityVec[e1][j] - relationVec[rel][j];
            if (L1_FLAG) {
                sum += Math.abs(tmp);
            } else {
                sum += tmp*tmp;
            }
        }
        return sum;
    }

    private void gradient_kb(int e1, int rel, int e2, double delta) {
        for (int j = 0; j < N; j++) {
            double x = 2 * (entityVec[e2][j] - entityVec[e1][j] - relationVec[rel][j]);
            if (L1_FLAG) {
                if (x > 0) {
                    x = 1;
                } else {
                    x = -1;
                }
            }

            synchronized (relationTmp[rel]) {
                relationTmp[rel][j] -= delta * RATE * x;
            }

            synchronized (entityTmp[e1]) {
                entityTmp[e1][j] -= delta * RATE * x;
            }

            synchronized (entityTmp[e2]) {
                entityTmp[e2][j] += delta * RATE * x;
            }
        }
    }

    private void addRelations(int headId, int tailId, int relationId,
            List<Pair<int[], Float>> pathResources) {
        Log.debug(headId + " " + tailId + " " + relationId + " " + pathResources.size());
        headIds.add(headId);
        tailIds.add(tailId);
        relationIds.add(relationId);
        fb_path.add(pathResources);
        Pair<Integer, Integer> key = new Pair<>(headId, relationId);
        positiveTriples.putIfAbsent(key, HashIntSets.newMutableSet());
        positiveTriples.get(key).add(tailId);
    }

    private void prepare() throws IOException {
        entityCount = 0;
        try (BufferedReader entityReader = FileTools.bufferedReader(KGCompletion.ENTITY2ID_FILE)) {
            for (String line = entityReader.readLine(); line != null; line = entityReader.readLine()) {
                List<String> split = WHITESPACE_SPLITTER.splitToList(line);
                String entity = split.get(0);
                int id = Integer.valueOf(split.get(1));
                entityToId.put(entity, id);
                entityCount++;
            }
        }

        try (BufferedReader relationReader = FileTools.bufferedReader(KGCompletion.RELATION2ID_FILE)) {
            for (String line = relationReader.readLine(); line != null; line = relationReader.readLine()) {
                // one for forward, one for inverse
                relationCount += 2;
            }
        }

        try (BufferedReader praReader = FileTools.bufferedReader(Pcra.Mode.TRAIN.getPraFile())) {
            for (String line = praReader.readLine(); line != null; line = praReader.readLine()) {
                Iterator<String> triple = WHITESPACE_SPLITTER.split(line).iterator();
                String head = triple.next();
                String tail = triple.next();
                int relationId = Integer.valueOf(triple.next());

                if (!entityToId.containsKey(head)) {
                    Log.error("PTransEAddTrain.prepare", "Missing entity " + head);
                    return;
                }
                if (!entityToId.containsKey(tail)) {
                    Log.error("PTransEAddTrain.prepare", "Missing entity " + tail);
                    return;
                }

                int headId = entityToId.getInt(head);
                int tailId = entityToId.getInt(tail);
                List<Pair<int[], Float>> b = new ArrayList<>();

                String pathResources = praReader.readLine();

                Iterator<String> parts = WHITESPACE_SPLITTER.split(pathResources).iterator();
                int size = Integer.valueOf(parts.next());
                for (int i = 0; i < size; i++) {
                    // Format is <path length> <path> <resource allocation>
                    int pathLength = Integer.valueOf(parts.next());
                    int[] path = new int[pathLength];
                    for (int j = 0; j < pathLength; j++) {
                        path[j] = Integer.valueOf(parts.next());
                    }
                    float resourceAllocation = Float.valueOf(parts.next());

                    b.add(new Pair<>(path, resourceAllocation));
                }

                if (parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected pathResources: " + pathResources);
                    return;
                }

                addRelations(headId, tailId, relationId, b);
            }
        }

        Log.info("relation_num=" + relationCount);
        Log.info("entity_num=" + entityCount);

        try (BufferedReader confidenceReader = FileTools.bufferedReader(Pcra.CONFIDENCE_FILE)) {
            for (String line = confidenceReader.readLine(); line != null; line = confidenceReader.readLine()) {
                Iterator<String> parts = WHITESPACE_SPLITTER.split(line).iterator();
                int size = Integer.valueOf(parts.next());

                List<String> pathList = new ArrayList<>();
                for (int i = 0; i < size; i++) {
                    pathList.add(parts.next());
                }
                String path = pathList.stream().collect(Collectors.joining(" "));

                if (parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected confidence path: " + line);
                    return;
                }

                String confidences = confidenceReader.readLine();

                Iterator<String> confidenceParts = WHITESPACE_SPLITTER.split(confidences).iterator();
                int confidenceLength = Integer.valueOf(confidenceParts.next());
                for (int i = 0; i < confidenceLength; i++) {
                    String relation = confidenceParts.next();

                    float confidence = Float.valueOf(confidenceParts.next());
                    pathConfidence.put(new Pair(path, relation), confidence);

                    Log.debug(path + " " + relation + " " + confidence);
                }

                if (parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected confidences: " + confidences);
                    return;
                }
            }
        }
    }

    static double vectorLength(float[] x) {
        double len = 0;
        for (int i = 0; i < x.length; i++) {
            len += x[i] * x[i];
        }
        return Math.sqrt(len);
    }

    private static void norm(float[] x) {
        synchronized (x) {
            double l = vectorLength(x);
            if (l > 1) {
                for (int i = 0; i < x.length; i++) {
                    x[i] /= l;
                }
            }
        }
    }

    private static double normal(double x, double mu, double sigma) {
        return 1.0 / Math.sqrt(2*Math.PI) / sigma * Math.exp(-1 *(x-mu)*(x-mu) / (2*sigma*sigma));
    }

    private static double rand(double min, double max) {
        return min + (max - min) * Math.random();
    }

    private static double randn(double mu, double sigma, double min, double max) {
        double x,y,dScope;
        do {
            x = rand(min, max);
            y = normal(x, mu, sigma);
            dScope = rand(0.0, normal(mu,mu,sigma));
        } while (dScope > y);
        return x;
    }

    private static int randMax(int max) {
        return (int) (Math.random() * max);
    }

    private static float[][] deepCopy(float[][] matrix) {
        return java.util.Arrays.stream(matrix).map(float[]::clone).toArray($ -> matrix.clone());
    }

    public static void main(String[] args) throws Exception {
        PTransEAddTrain add = new PTransEAddTrain();
        add.train();
    }
}
