package com.diffbot.ml;

import com.diffbot.toolbox.FileTools;
import com.esotericsoftware.minlog.Log;
import com.google.common.base.Splitter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Java implementation of Path-Constraint Resource Allocation.
 *
 * See Wang et al, Knowledge Graph and Text Jointly Embedding, 2014 and
 * https://github.com/thunlp/KB2E/tree/master/PTransE
 *
 * Inputs (provided by KGCompletion):
 *     train.txt: training triples like `e1 e2 rel`
 *     test.txt: test triples, same as above
 *     entity2id.txt: entities and corresponding ids like `Qu2aSs1 145223`
 *     relation2id.txt: relations, same as above
 *
 * Outputs:
 *     path2.txt
 *     confidence.txt
 *     train_pra.txt
 *     test_pra.txt
 *
 * TODO: reimplement in Spark/Tensorflow
 */
public class Pcra {
    private static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("#0.0000");
    private static final int LOG_FREQUENCY = 1_000;
    private static final float MIN_RESOURCE = 0.01f;
    private static final Splitter WHITESPACE_SPLITTER = Splitter.onPattern("\\s+").trimResults();
    private static final File TEST_PRA_FILE = new File(KGCompletion.KB2E_DIRECTORY, "test_pra.txt");
    private static final File TRAIN_PRA_FILE = new File(KGCompletion.KB2E_DIRECTORY, "train_pra.txt");

    static final Splitter PATH_SPLITTER = Splitter.onPattern(">").trimResults();
    static final File PATH2_FILE = new File(KGCompletion.KB2E_DIRECTORY, "path2.txt");
    static final File CONFIDENCE_FILE = new File(KGCompletion.KB2E_DIRECTORY, "confidence.txt");

    protected enum Mode {
        TEST {
            @Override public File getTriplesFile() {
                return KGCompletion.TEST_FILE;
            }

            @Override public File getPathResourceFile() {
                return TEST_PRA_FILE;
            }
        },
        TRAIN {
            @Override public File getTriplesFile() {
                return KGCompletion.TRAIN_FILE;
            }

            @Override public File getPathResourceFile() {
                return TRAIN_PRA_FILE;
            }
        },
        ;

        public abstract File getTriplesFile();
        public abstract File getPathResourceFile();
    }

    /**
     * Attempts to determine the reliability of 2-step relation paths by allocating a fixed budget
     * of `resource` to flow from each relation to its co-occurring relations and maintaining
     * the paths with > MIN_RESOURCE.
     *
     * For example, BornInState->StateInCountry implies Nationality with high reliability but
     * Friend->Profession does not generalize and should have low resource and low reliability.
     *
     * See Lin et al, Modeling Relation Paths for Representation Learning of Knowledge Bases, 2015
     *
     * TODO: parallelize
     */
    private static void pathConstraintResourceAllocation() throws IOException {
        long startMs = System.currentTimeMillis();

        Log.info("PCRA", "Loading relations...");
        Map<String, String> relationToId = getRelationIds();
        int relationCount = relationToId.size();
        Log.info("PCRA", String.format("Loaded %d relations in %dms.",
                relationCount, System.currentTimeMillis() - startMs));

        // TODO: use Dgraph
        // map of `headId tailId` -> relationId's
        Map<String, Set<String>> pairRelationMap = new HashMap<>();
        // map of `headId` -> (map of `relationId` -> tailId's)
        Map<String, Map<String, Set<String>>> headRelationTailMap = new HashMap<>();
        // map of `headId tailId` -> map of (relationId -> resource)
        Map<String, Map<String, Float>> pathResources = new HashMap<>();

        startMs = System.currentTimeMillis();
        Log.info("PCRA", "Loading training relations...");
        int trainingTriples = addTrainingRelations(pairRelationMap, headRelationTailMap,
                relationToId, relationCount);
        Log.info("PCRA", String.format("Loaded %d training relations in %dms.",
                trainingTriples, System.currentTimeMillis() - startMs));

        // counts of relation occurrences
        Map<String, Integer> pathCounts = new HashMap<>();
        // counts of relation co-occurrences
        Map<String, Integer> relatedPathCounts = new HashMap<>();

        // TODO: write directly to file, keep sum
        // relations with > MIN_RESOURCE
        Set<String> reliablePaths = new HashSet<>();

        startMs = System.currentTimeMillis();
        Log.info("PCRA", "Executing Path-Constraint Resource Allocation...");

        // Count 1-hop paths
        // TODO: parallelize
        int countedHeadEntities = 0;
        for (String headId : headRelationTailMap.keySet()) {
            countedHeadEntities++;
            for (String relationId : headRelationTailMap.get(headId).keySet()) {
                Set<String> tailIds = headRelationTailMap.get(headId).get(relationId);
                for (String tailId : tailIds) {
                    // Count relation frequencies
                    mapIncrement(pathCounts, relationId);

                    // Count frequency for this particular head,tail pair
                    String entityPairKey = getEntityPairKey(headId, tailId);
                    for (String nestedRelationId : pairRelationMap.get(entityPairKey)) {
                        mapIncrement(relatedPathCounts, getPathKey(relationId, nestedRelationId));
                    }

                    // Flow resource from head to tail
                    float delta = 1 / (float) tailIds.size();
                    nestedMapAdd(pathResources, entityPairKey, String.valueOf(relationId), delta);
                }
            }

            if (countedHeadEntities % LOG_FREQUENCY == 0) {
                Log.info("PCRA", String.format("\tCounted 1-hop path frequencies from %d "
                        + "of %d source entities in time: %dms", countedHeadEntities,
                        headRelationTailMap.size(), System.currentTimeMillis() - startMs));
            }
        }

        System.out.println("PathCounts: " + pathCounts.size());
        System.out.println("RelatedPaths: " + relatedPathCounts.size());

        // TODO: parallelize
        countedHeadEntities = 0;
        for (String headId : headRelationTailMap.keySet()) {
            countedHeadEntities++;
            // Count 2-hop paths and flow resources from 1-hop parent paths
            for (String relationId : headRelationTailMap.get(headId).keySet()) {
                Set<String> tailIds = headRelationTailMap.get(headId).get(relationId);
                for (String tailId : tailIds) {
                    String entityPairKey = getEntityPairKey(headId, tailId);

                    // Traverse relations sourced from the tail (2-hops from the original head)
                    if (headRelationTailMap.containsKey(tailId)) {
                        for (String nestedRelationId : headRelationTailMap.get(tailId).keySet()) {
                            Set<String> tailTailIds = headRelationTailMap.get(tailId).get(nestedRelationId);
                            String nestedRelationKey = getPathKey(relationId, nestedRelationId);
                            for (String tailTailId : tailTailIds) {
                                mapIncrement(pathCounts, nestedRelationKey);

                                String nestedPairKey = getEntityPairKey(headId, tailTailId);
                                if (pairRelationMap.containsKey(nestedPairKey)) {
                                    for (String key : pairRelationMap.get(nestedPairKey)) {
                                        mapIncrement(relatedPathCounts, getPathKey(nestedRelationKey, key));
                                    }

                                    // Flow existing resource from the 1-hop tail to 2-hop tails
                                    float delta = pathResources.get(entityPairKey).get(relationId) / (float) tailTailIds.size();
                                    nestedMapAdd(pathResources, nestedPairKey, nestedRelationKey, delta);
                                }
                            }
                        }
                    }
                }
            }

            if (countedHeadEntities % LOG_FREQUENCY == 0) {
                Log.info("PCRA", String.format("\tCounted 2-hop path frequencies from %d " +
                                "of %d source entities in time: %dms", countedHeadEntities,
                        headRelationTailMap.size(), System.currentTimeMillis() - startMs));
            }
        }

        System.out.println("PathCounts: " + pathCounts.size());
        System.out.println("RelatedPaths: " + relatedPathCounts.size());

        try (BufferedWriter path2Writer = FileTools.bufferedWriter(PATH2_FILE)) {
            // TODO: parallelize
            countedHeadEntities = 0;
            for (String headId : headRelationTailMap.keySet()) {
                countedHeadEntities++;
                // Check all entities as possible tail or 2-hop tails
                for (String tailId : headRelationTailMap.keySet()) {
                    String entityPairKey = getEntityPairKey(headId, tailId);
                    if (pathResources.containsKey(entityPairKey)) {
                        path2Writer.write(entityPairKey + "\n");

                        Map<String, Float> matchingPathResources = new HashMap<>();
                        float sum = 0;
                        for (Map.Entry<String, Float> entry : pathResources.get(entityPairKey).entrySet()) {
                            matchingPathResources.put(entry.getKey(), entry.getValue());
                            sum += entry.getValue();
                        }

                        Map<String, Float> reliablePathResources = new HashMap<>();
                        for (String key : matchingPathResources.keySet()) {
                            if (matchingPathResources.get(key) / sum > MIN_RESOURCE) {
                                reliablePathResources.put(key, matchingPathResources.get(key) / sum);
                            }
                        }

                        path2Writer.write(String.valueOf(reliablePathResources.size()));
                        for (String path : reliablePathResources.keySet()) {
                            reliablePaths.add(path);
                            int length = PATH_SPLITTER.splitToList(path).size();
                            path2Writer.write(" " + length + " " + path + " " + DECIMAL_FORMAT.format(reliablePathResources.get(path)));
                        }
                        path2Writer.write("\n");
                    }
                }

                if (countedHeadEntities % LOG_FREQUENCY == 0) {
                    Log.info("PCRA", String.format("\tScored path resources from %d " +
                                    "of %d source entities in time: %dms",
                            countedHeadEntities, headRelationTailMap.size(),
                            System.currentTimeMillis() - startMs));
                }
            }
        }
        Log.info("PCRA", String.format("Executed Path-Constraint Resource "
                        + "Allocation in %dms.", System.currentTimeMillis() - startMs));

        Log.info("PCRA", "Saving path confidences to " + CONFIDENCE_FILE + "...");
        try (BufferedWriter confidenceWriter = FileTools.bufferedWriter(CONFIDENCE_FILE)) {
            for (String path : reliablePaths) {
                List<String> out = new ArrayList<>();
                for (int i = 0; i < relationCount; i++) {
                    String pathKey = getPathKey(path, String.valueOf(i));
                    if (pathCounts.containsKey(path) && relatedPathCounts.containsKey(pathKey)) {
                        float confidence = relatedPathCounts.get(pathKey) / (float) pathCounts.get(path);
                        out.add(" " + i + " " + DECIMAL_FORMAT.format(confidence));
                    }
                }
                if (!out.isEmpty()) {
                    confidenceWriter.write("" + PATH_SPLITTER.splitToList(path).size() + " " + path + "\n");
                    confidenceWriter.write(String.valueOf(out.size()));
                    for (String o : out) {
                        confidenceWriter.write(o);
                    }
                    confidenceWriter.write("\n");
                }
            }
        }

        writePaths(Mode.TRAIN, relationToId, relationCount, pathResources);
        writePaths(Mode.TEST, relationToId, relationCount, pathResources);
    }

    private static Map<String, String> getRelationIds() throws IOException {
        Map<String, String> relationToId = new HashMap<>();
        try (BufferedReader relationReader = FileTools.bufferedReader(KGCompletion.RELATION2ID_FILE)) {
            for (String line = relationReader.readLine(); line != null; line = relationReader.readLine()) {
                Iterator<String> split = WHITESPACE_SPLITTER.split(line).iterator();
                String headId = split.next();
                String tailId = split.next();
                relationToId.put(headId, tailId);
            }
        }
        return relationToId;
    }

    // TODO: convert to database writes
    private static int addTrainingRelations(Map<String, Set<String>> pairRelationMap,
            Map<String, Map<String, Set<String>>> headRelationTailMap,
            Map<String, String> relationToId, int relationCount) throws IOException {
        long start = System.currentTimeMillis();
        int tripleCount = 0;
        try (BufferedReader trainReader = FileTools.bufferedReader(KGCompletion.TRAIN_FILE)) {
            for (String line = trainReader.readLine(); line != null; line = trainReader.readLine()) {
                Iterator<String> split = WHITESPACE_SPLITTER.split(line).iterator();
                String headId = split.next();
                String tailId = split.next();
                String relationId = relationToId.get(split.next());
                // TODO: ignore inverse relations?
                String inverseRelationId = getInverseRelationId(relationId, relationCount);
                tripleCount += 2;

                String tripleKey = getEntityPairKey(headId, tailId);
                pairRelationMap.putIfAbsent(tripleKey, new HashSet<>());
                pairRelationMap.get(tripleKey).add(relationId);

                String reverseTripleKey = getEntityPairKey(tailId, headId);
                pairRelationMap.putIfAbsent(reverseTripleKey, new HashSet<>());
                pairRelationMap.get(reverseTripleKey).add(inverseRelationId);

                headRelationTailMap.putIfAbsent(headId, new HashMap<>());
                headRelationTailMap.get(headId).putIfAbsent(relationId, new HashSet<>());
                headRelationTailMap.get(headId).get(relationId).add(tailId);

                headRelationTailMap.putIfAbsent(tailId, new HashMap<>());
                headRelationTailMap.get(tailId).putIfAbsent(inverseRelationId, new HashSet<>());
                headRelationTailMap.get(tailId).get(inverseRelationId).add(headId);

                if (relationCount % 100_000 == 0) {
                    Log.info("Pcra.addTrainingRelations", "Loaded " +
                            tripleCount + " triples in " + (System.currentTimeMillis() - start) + "ms...");
                }
            }
        }
        return tripleCount;
    }

    private static void writePaths(Mode mode, Map<String, String> relationToId, int relationCount,
            Map<String, Map<String, Float>> pathResources) throws IOException {
        try (BufferedReader br = FileTools.bufferedReader(mode.getTriplesFile());
                BufferedWriter bw = FileTools.bufferedWriter(mode.getPathResourceFile())) {
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                Iterator<String> split = WHITESPACE_SPLITTER.split(line).iterator();
                String head = split.next();
                String tail = split.next();
                String relationId = relationToId.get(split.next());

                writePaths(bw, head, tail, relationId, pathResources);
                writePaths(bw, tail, head, getInverseRelationId(relationId, relationCount),
                        pathResources);
            }
        }
    }

    private static void writePaths(BufferedWriter bw, String head, String tail,
            String relationId, Map<String, Map<String, Float>> pathResources) throws IOException {
        bw.write(String.format("%s %s %s\n", head, tail, relationId));

        Map<String, Float> matchingPathResources = new HashMap<>();
        Map<String, Float> reliablePathResources = new HashMap<>();

        String entityPairKey = getEntityPairKey(head, tail);
        if (pathResources.containsKey(entityPairKey)) {
            float sum = 0;
            for (String path : pathResources.get(entityPairKey).keySet()) {
                matchingPathResources.put(path, pathResources.get(entityPairKey).get(path));
                sum += matchingPathResources.get(path);
            }
            for (String path : matchingPathResources.keySet()) {
                matchingPathResources.put(path, matchingPathResources.get(path) / sum);
                if (matchingPathResources.get(path) > MIN_RESOURCE) {
                    reliablePathResources.put(path, matchingPathResources.get(path));
                }
            }
        }

        bw.write(String.valueOf(reliablePathResources.size()));
        for (String path : reliablePathResources.keySet()) {
            bw.write(" " + PATH_SPLITTER.splitToList(path).size() + " " + path + " " +
                    DECIMAL_FORMAT.format(reliablePathResources.get(path)));
        }
        bw.write("\n");
    }

    private static String getEntityPairKey(String headId, String tailId) {
        return headId + " " + tailId;
    }

    private static String getPathKey(String relation1, String relation2) {
        return "" + relation1 + ">" + relation2;
    }

    private static String getInverseRelationId(String relationId, int relationCount) {
        int relationInt = Integer.valueOf(relationId);
        return relationInt < relationCount ? String.valueOf(relationInt + relationCount) :
                String.valueOf(relationInt - relationCount);
    }

    private static void nestedMapAdd(Map<String, Map<String, Float>> map, String key1,
            String key2, float delta) {
        map.putIfAbsent(key1, new HashMap<>());
        map.get(key1).compute(key2, (k, v) -> v == null ? delta : v + delta);
    }

    private static void mapIncrement(Map<String, Integer> map, String key) {
        map.compute(key, (k, v) -> v == null ? 1 : v + 1);
    }

    public static void main(String[] args) throws Exception {
        pathConstraintResourceAllocation();
    }
}
