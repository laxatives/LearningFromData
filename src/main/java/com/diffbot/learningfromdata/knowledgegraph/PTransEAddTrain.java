package com.diffbot.ml;

import com.diffbot.thoth.utilities.Triple;
import com.diffbot.toolbox.FileTools;
import com.diffbot.utils.Pair;
import com.esotericsoftware.minlog.Log;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.AtomicDouble;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Combines multi-hop paths using vector addition. This appears to be the best scoring operator in
 * {ADD, MUL, RNN}.
 *
 * Inputs (provided by Pcra):
 *     entity2id.txt
 *     relation2id.txt
 *     train_pra.txt
 *     confidence.txt
 *
 * Outputs:
 *     relation2vec.txt
 *     entity2vec.txt
 *
 * TODO: port to Spark/Tensorflow
 */
public class PTransEAddTrain {
    private static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("#0.000000");
    private static final Splitter WHITESPACE_SPLITTER = Splitter.onPattern("\\s+").trimResults().omitEmptyStrings();
    private static final Random RANDOM = new Random();
    // TODO: tune rate for Diffbot KG
    private static final float INITIAL_LEARNING_RATE = 0.01f;
    private static final float MARGIN = 1f;
    private static final int BATCH_COUNT = 100;
    private static final int EPOCHS = 1000;

    /**
     * The degree of the latent feature space.
     */
    static final int N = 25;
    static final File ENTITY2VEC_FILE = new File(KGCompletion.KB2E_DIRECTORY, "entity2vec.txt");
    static final File RELATION2VEC_FILE = new File(KGCompletion.KB2E_DIRECTORY, "relation2vec.txt");

    private int entityCount = 0;
    private int relationCount = 0;
    private Map<String, Integer> entityToId = new HashMap<>();
    private List<Integer> headIds = new ArrayList<>();
    private List<Integer> tailIds = new ArrayList<>();
    private List<Integer> relationIds = new ArrayList<>();
    private List<List<Pair<int[], Float>>> pathResources = new ArrayList<>();
    private Map<Pair<int[], Integer>, Float> pathConfidences = new HashMap<>();

    /**
     * A map of Pair<headId, relationId> -> Set<tailId> where every headId, readId, tailId triple
     * implies a positive label.
     */
    private Map<Pair<Integer, Integer>, Set<Integer>> positiveTriples = new HashMap<>();

    private float learningRate = INITIAL_LEARNING_RATE;
    private AtomicDouble epochError;
    private float[][] entityTmp;
    private float[][] entityVec;
    private float[][] relationTmp;
    private float[][] relationVec;

    public void train() throws IOException {
        prepare();
        run();
    }

    private void run() throws IOException {
        Log.info("PTransEAddTrain.run", "N=" + N);

        entityVec = new float[entityCount][N];
        for (int i = 0; i < entityCount; i++) {
            for (int j = 0; j < N; j++) {
                entityVec[i][j] = (float) RANDOM.nextGaussian();
            }
            normalizeL2(entityVec[i]);
        }

        relationVec = new float[relationCount][N];
        for (int i = 0; i< relationCount; i++) {
            for (int j = 0; j < N; j++) {
                relationVec[i][j] = (float) RANDOM.nextGaussian();
            }
            normalizeL2(relationVec[i]);
        }

        bfgs();
        writeRelationVec();
        writeEntityVec();
    }

    /**
     * Broyden–Fletcher–Goldfarb–Shanno is a quasi-Newton Method for gradient-descent.
     */
    private void bfgs() throws IOException {
        long startMs = System.currentTimeMillis();
        Log.info("PTransEAddTrain.bfgs", "margin=" + MARGIN);
        int batchSize = headIds.size() / BATCH_COUNT;

        relationTmp = deepCopy(relationVec);
        entityTmp = deepCopy(entityVec);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            epochError = new AtomicDouble();
            if (epoch % (EPOCHS / 5) == 0) {
                if (epoch > 0) {
                    learningRate /= 2;
                }
                Log.info("PTransEAddTrain.bfgs", "learningRate=" + learningRate);
            }

            for (int batch = 0; batch < BATCH_COUNT; batch++) {
                IntStream.range(0, batchSize).parallel().forEach(k -> {
                    int i = randMax(headIds.size());

                    int headId = headIds.get(i);
                    int tailId = tailIds.get(i);
                    int relationId = relationIds.get(i);

                    // For every positive label, generate a negative label through corruption
                    Triple<Integer, Integer, Integer> corruptTriple =
                            corruptTriple(headId, relationId, tailId);
                    int corruptHeadId = corruptTriple.first;
                    int corruptRelationId = corruptTriple.second;
                    int corruptTailId = corruptTriple.third;

                    trainKb(headId, relationId, tailId, corruptHeadId, corruptRelationId, corruptTailId);

                    Set<Integer> modifiedRelationIds = Sets.newHashSet(relationId, corruptRelationId);
                    if (!pathResources.get(i).isEmpty()) {
                        int corruptPathRelationId = randMax(relationCount);
                        while (positiveTripleExists(new Pair<>(headId, corruptPathRelationId), tailId)) {
                            corruptPathRelationId = randMax(relationCount);
                        }

                        for (Pair<int[], Float> pathConfidencePair : pathResources.get(i)) {
                            int[] path = pathConfidencePair.first;
                            double pathResource = pathConfidencePair.second;
                            Arrays.stream(path).forEach(modifiedRelationIds::add);

                            float pathConfidence = Math.max(0.01f, pathConfidences.getOrDefault(
                                    new Pair<>(path, relationId), 0f));
                            trainPath(relationId, corruptPathRelationId, path,
                                    2 * MARGIN, pathResource * pathConfidence);
                        }
                    }

                    // TODO: these aren't really thread safe with train/update
                    for (int id : ImmutableSet.of(headId, tailId, corruptHeadId, corruptTailId)) {
                        synchronized(entityTmp[id]) {
                            normalizeL2(entityTmp[id]);
                        }
                    }
                    // TODO: the reference doesn't normalize corruptRelationIds or paths?
                    for (int id : modifiedRelationIds) {
                        synchronized(relationTmp[id]) {
                            normalizeL2(relationTmp[id]);
                        }
                    }

                });

                relationVec = deepCopy(relationTmp);
                entityVec = deepCopy(entityTmp);
            }

            if (epoch % 10 == 0) {
                Log.info("PTransEAddTrain.bfgs",
                        "epoch " + epoch + "/" + EPOCHS + ", epochError=" +
                                epochError.get() + ", time=" +
                                (System.currentTimeMillis() - startMs) + "ms");
            }
        }
    }

    private void writeRelationVec() throws IOException {
        try (BufferedWriter relationVecWriter = FileTools.bufferedWriter(RELATION2VEC_FILE)) {
            for (int i = 0; i < relationCount; i++) {
                normalizeL2(relationVec[i]);
                for (int j = 0; j < N; j++) {
                    relationVecWriter.write(DECIMAL_FORMAT.format(relationVec[i][j]) + "\t");
                }
                relationVecWriter.write("\n");
            }
        }
    }

    private void writeEntityVec() throws IOException {
        try (BufferedWriter entityVecWriter = FileTools.bufferedWriter(ENTITY2VEC_FILE)) {
            for (int i = 0; i < entityCount; i++) {
                normalizeL2(entityVec[i]);
                for (int j = 0; j < N; j++) {
                    entityVecWriter.write(DECIMAL_FORMAT.format(entityVec[i][j]) + "\t");
                }
                entityVecWriter.write("\n");
            }
        }
    }

    /**
     * Generates a negative label by corrupting the provided positive label by randomly changing one
     * of {headId, relationId, tailId} such that the new triple is not in the positive label set.
     */
    private Triple<Integer, Integer, Integer> corruptTriple(int headId, int relationId, int tailId) {
        double tmp = Math.random();
        if (tmp < 0.25) {
            // Corrupt the head entity
            int corruptEntityId = randMax(entityCount);
            while (positiveTripleExists(new Pair<>(corruptEntityId, relationId), tailId)) {
                corruptEntityId = randMax(entityCount);
            }

            return new Triple<>(corruptEntityId, relationId, tailId);
        } else if (tmp < 0.5) {
            // Corrupt the tail entity
            Pair<Integer, Integer> key = new Pair<>(headId, relationId);

            int corruptEntityId = randMax(entityCount);
            while (positiveTripleExists(key, corruptEntityId)) {
                corruptEntityId = randMax(entityCount);
            }
            return new Triple<>(headId, relationId, corruptEntityId);
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

    /**
     * Attempts to minimize the difference between highly correlated relations and composed paths.
     */
    private void trainPath(int relationId, int corruptRelationId, int[] path, double margin,
            double delta) {
        double trainError = evalPath(relationId, path);
        double corruptError = evalPath(corruptRelationId, path);
        if (trainError + margin > corruptError) {
            epochError.addAndGet(delta * (margin + trainError - corruptError));
            updateRelation(relationId, path, -delta);
            updateRelation(corruptRelationId, path, delta);
        }
    }

    private double evalPath(int r1, int[] rel_path) {
        double error = 0;
        for (int k = 0; k < N; k++) {
            double tmp = relationVec[r1][k];
            for (int j : rel_path) {
                tmp -= relationVec[j][k];
            }

            error += Math.abs(tmp);
        }
        return error;
    }

    private void updateRelation(int r1, int[] rel_path, double delta) {
        for (int k = 0; k < N; k++) {
            double x = relationVec[r1][k];
            for (int path : rel_path) {
                x -= relationVec[path][k];
            }

            relationTmp[r1][k] += delta * learningRate * x;

            x /= (float) rel_path.length;
            for (int path : rel_path) {
                relationTmp[path][k] -= delta * learningRate * x;
            }
        }
    }

    /**
     * Compute a hinge loss on the true label and the corrupt label and perform gradient descent if
     * necessary.
     */
    private void trainKb(int trueHeadId, int trueRelationId, int trueTailId,
            int corruptHeadId, int corruptRelationId, int corruptTailId) {
        double positiveError = evalTriple(trueHeadId, trueTailId, trueRelationId);
        double corruptError = evalTriple(corruptHeadId, corruptTailId, corruptRelationId);
//        System.out.println(trueHeadId+","+trueTailId+","+trueRelationId+";"+corruptHeadId+","+corruptTailId+","+corruptRelationId+" -> "+positiveScore+":"+corruptScore);
        if (positiveError + MARGIN > corruptError) {
            epochError.addAndGet(positiveError + MARGIN - corruptError);
            updateTriple(trueHeadId, trueRelationId, trueTailId, -1);
            updateTriple(corruptHeadId, corruptRelationId, corruptTailId, 1);
        }
    }

    private double evalTriple(int e1,int e2,int rel) {
        double error = 0;
        for (int j = 0; j < N; j++) {
            double err = entityVec[e2][j] - entityVec[e1][j] - relationVec[rel][j];
            error += Math.abs(err);
        }
        return error;
    }

    private void updateTriple(int headId, int relationId, int tailId, double delta) {
        for (int j = 0; j < N; j++) {
            double x = entityVec[tailId][j] - entityVec[headId][j] - relationVec[relationId][j];

            relationTmp[relationId][j] -= delta * learningRate * x;
            entityTmp[headId][j] -= delta * learningRate * x;
            entityTmp[tailId][j] += delta * learningRate * x;
        }
    }

    /**
     * Adds a triple and any associated paths from PCRA to the labelset.
     *
     * These are strictly positive labels.
     */
    private void addLabel(int headId, int tailId, int relationId,
            List<Pair<int[], Float>> pathResources) {
        Log.debug(headId + " " + tailId + " " + relationId + " " + pathResources.size());
        headIds.add(headId);
        tailIds.add(tailId);
        relationIds.add(relationId);
        this.pathResources.add(pathResources);
        Pair<Integer, Integer> key = new Pair<>(headId, relationId);
        positiveTriples.putIfAbsent(key, new HashSet<>());
        positiveTriples.get(key).add(tailId);
    }

    private void prepare() throws IOException {
        Log.info("PTransEAddTrain.prepare", "Loading entities from " +
                KGCompletion.ENTITY2ID_FILE + "...");
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

        Log.info("PTransEAddTrain.prepare", "Loading relations from " +
                KGCompletion.RELATION2ID_FILE + "...");
        try (BufferedReader relationReader = FileTools.bufferedReader(KGCompletion.RELATION2ID_FILE)) {
            for (String line = relationReader.readLine(); line != null; line = relationReader.readLine()) {
                // one for forward, one for inverse
                relationCount += 2;
            }
        }

        Log.info("PTransEAddTrain.prepare", "Loading triples from " +
                Pcra.Mode.TRAIN.getPathResourceFile() + "...");
        try (BufferedReader praReader = FileTools.bufferedReader(Pcra.Mode.TRAIN.getPathResourceFile())) {
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

                int headId = entityToId.get(head);
                int tailId = entityToId.get(tail);
                List<Pair<int[], Float>> pathResources = new ArrayList<>();

                String pathLine = praReader.readLine();

                Iterator<String> parts = WHITESPACE_SPLITTER.split(pathLine).iterator();
                int size = Integer.valueOf(parts.next());
                for (int i = 0; i < size; i++) {
                    // Format is <path length> <path> <resource allocation>
                    int pathLength = Integer.valueOf(parts.next());
                    int[] path = new int[pathLength];
                    Iterator<String> pathParts = Pcra.PATH_SPLITTER.split(parts.next()).iterator();
                    for (int j = 0; j < pathLength; j++) {
                        try {
                            path[j] = Integer.valueOf(pathParts.next());
                        } catch (NumberFormatException e) {
                            Log.error("PTransEAddTrain.prepare",
                                    "Invalid path resource file row:\n" + line + "\n" + pathLine, e);
                            throw e;
                        }
                    }

                    if (pathParts.hasNext()) {
                        Log.error("PTransEAddTrain.prepare", "Unexpected pathResources: " + pathLine);
                        return;
                    }

                    float resourceAllocation = Float.valueOf(parts.next());

                    pathResources.add(new Pair<>(path, resourceAllocation));
                }

                if (parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected pathResources: " + pathLine);
                    return;
                }

                addLabel(headId, tailId, relationId, pathResources);
            }
        }
        Log.info("PTransEAddTrain.prepare",
                "\tentityCount=" + entityCount + ", relationCount=" + relationCount);

        Log.info("PTransEAddTrain.prepare", "Loading paths from " +
                Pcra.CONFIDENCE_FILE + "...");
        try (BufferedReader confidenceReader = FileTools.bufferedReader(Pcra.CONFIDENCE_FILE)) {
            for (String line = confidenceReader.readLine(); line != null; line = confidenceReader.readLine()) {
                Iterator<String> parts = WHITESPACE_SPLITTER.split(line).iterator();
                int size = Integer.valueOf(parts.next());

                int[] path = new int[size];
                Iterator<String> pathParts = Pcra.PATH_SPLITTER.split(parts.next()).iterator();
                for (int i = 0; i < size; i++) {
                    path[i] = Integer.valueOf(pathParts.next());
                }

                if (pathParts.hasNext() || parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected confidence path: " + line);
                    return;
                }

                String confidences = confidenceReader.readLine();

                Iterator<String> confidenceParts = WHITESPACE_SPLITTER.split(confidences).iterator();
                int confidenceLength = Integer.valueOf(confidenceParts.next());
                for (int i = 0; i < confidenceLength; i++) {
                    int relation = Integer.valueOf(confidenceParts.next());

                    float confidence = Float.valueOf(confidenceParts.next());
                    pathConfidences.put(new Pair<>(path, relation), confidence);

                    Log.debug(Arrays.toString(path) + " " + relation + " " + confidence);
                }

                if (parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected confidences: " + confidences);
                    return;
                }
            }
        }
    }

    public static double l2Norm(float[] x) {
        double len = 0;
        for (float i : x) {
            len += i * i;
        }
        return Math.sqrt(len);
    }

    private static void normalizeL2(float[] x) {
        double l = l2Norm(x);
        if (l > 1) {
            for (int i = 0; i < x.length; i++) {
                x[i] /= l;
            }
        }
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
