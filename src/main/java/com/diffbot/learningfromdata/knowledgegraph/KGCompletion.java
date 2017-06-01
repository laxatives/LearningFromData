package com.diffbot.ml;

import com.diffbot.entities.DiffbotEntity;
import com.diffbot.kg.knowledgefusion.SecondPassKnowledgeFusion;
import com.diffbot.toolbox.FileTools;
import com.diffbot.utils.DiffbotEntityReader;
import com.esotericsoftware.minlog.Log;
import org.ojalgo.array.ArrayAnyD;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class KGCompletion {
    private static final File ENTITY_DUMP = new File(
            "/mnt/raid1/Thoth2/data/bleeding_edge/common/knowledgeFusionSecondPass",
            SecondPassKnowledgeFusion.FINAL_OUTPUT_PATH);
    private static final int MINIMUM_RELATIONS = 4;

    public static final File KB2E_DIRECTORY = new File("data/kb2e");

    static final File TRAIN_FILE = new File(KB2E_DIRECTORY, "train.txt");
    static final File VALID_FILE = new File(KB2E_DIRECTORY, "valid.txt");
    static final File TEST_FILE = new File(KB2E_DIRECTORY, "test.txt");
    static final File ENTITY2ID_FILE = new File(KB2E_DIRECTORY, "entity2id.txt");
    static final File RELATION2ID_FILE = new File(KB2E_DIRECTORY, "relation2id.txt");
    static final File E1_E2_FILE = new File(KB2E_DIRECTORY, "e1_e2.txt");

    /**
     * Creates an NxNxR adjacency tensor where
     * N is the number of selected entities (based on getAdjacencyMap) and
     * R is the number of relations (enum KGRelation).
     *
     * Each NxN slice is a symmetric [0,1] matrix.
     */
    private static void createNewTensor(boolean symmetric) {
        long start = System.currentTimeMillis();

        Map<String, List<Set<String>>> adjacencyMap = getAdjacencyMap();

        Set<String> entityIds = new HashSet<>(adjacencyMap.keySet());
        // TODO: only add targetIds if they are (targeted with frequency >= MINIMUM_RELATIONS)
        adjacencyMap.values().forEach(l -> l.forEach(entityIds::addAll));
        List<String> entityIndex = new ArrayList<>(entityIds);
        int entityCount = entityIndex.size();

        long durationMs = System.currentTimeMillis() - start;
        Log.info("KGCompletion.createNewTensor",
                String.format("Created adjacencyMap with %d entries and %d entities in %dms.",
                        adjacencyMap.keySet().size(), entityCount, durationMs));

        int depth = KGRelation.values().length;
        // TODO: try complex, see https://arxiv.org/pdf/1702.06879.pdf
        ArrayAnyD<Double> tensor = ArrayAnyD.DIRECT32.makeZero(entityCount, entityCount, depth);
        for (int i = 0; i < depth; i++) {
            for (Map.Entry<String, List<Set<String>>> entry : adjacencyMap.entrySet()) {
                if (!entityIndex.contains(entry.getKey())) {
                    continue;
                }
                int sourceIndex = entityIndex.indexOf(entry.getKey());

                Set<String> targetEntityIds = entry.getValue().get(i);
                for (String entityId : targetEntityIds) {
                    if (!entityIndex.contains(entityId)) {
                        continue;
                    }
                    int targetIndex = entityIndex.indexOf(entityId);
                    tensor.set(new long[]{sourceIndex, targetIndex, i}, 1);

                    if (symmetric) {
                        tensor.set(new long[]{targetIndex, sourceIndex, i}, 1);
                    }
                }
            }
        }
    }

    /**
     * Creates the inputs for com.diffbot.ml.TransE and other embeddings in
     * https://github.com/thunlp/KB2E
     *
     * Creates the following files in KB2E_DIRECTORY:
     * train.txt: training file, format (e1, e2, rel).
     * valid.txt: validation file, same format as train.txt
     * test.txt: test file, same format as train.txt.
     * entity2id.txt: all entities and corresponding ids, one per line.
     * relation2id.txt: all relations and corresponding ids, one per line.
     * e1_e2.txt: all top-500 entity pairs mentioned in the task entity prediction.
     */
    private static void createKb2eInputFiles() throws IOException {
        final float TEST_HOLDOUT = 0.1f;
        final float VALID_HOLDOUT = 0.1f;

        KB2E_DIRECTORY.mkdirs();

        long start = System.currentTimeMillis();
        Map<String, List<Set<String>>> adjacencyMap = getAdjacencyMap();

        Set<String> entityIds = new HashSet<>(adjacencyMap.keySet());
        // TODO: only add targetIds if they are (targeted with frequency >= MINIMUM_RELATIONS)
        adjacencyMap.values().forEach(l -> l.forEach(entityIds::addAll));
        List<String> entityIndex = new ArrayList<>(entityIds);
        int entityCount = entityIndex.size();

        long durationMs = System.currentTimeMillis() - start;
        Log.info("KGCompletion.createKb2eInputFiles",
                String.format("Created adjacencyMap with %d entries and %d entities in %dms.",
                        adjacencyMap.keySet().size(), entityCount, durationMs));

        Log.info("KGCompletion.createKb2eInputFiles",
                "Writing {train/valid/test}.txt...");
        try (BufferedWriter trainWriter =
                FileTools.bufferedWriter(TRAIN_FILE);
                BufferedWriter validWriter =
                        FileTools.bufferedWriter(VALID_FILE);
                BufferedWriter testWriter =
                        FileTools.bufferedWriter(TEST_FILE)) {
            for (Map.Entry<String, List<Set<String>>> entry : adjacencyMap.entrySet()) {
                String headId = entry.getKey();
                if (!entityIndex.contains(headId)) {
                    continue;
                }

                for (int i = 0; i < KGRelation.values().length; i++) {
                    String relation = KGRelation.values()[i].name();
                    for (String tailId : entry.getValue().get(i)) {
                        if (!entityIndex.contains(tailId)) {
                            continue;
                        }
                        /**
                         * TODO: First,  we  remove  the  user  profiles, version control, and meta
                         *  data, leaving 52,124,755 entities, 4,490 relations, and 204,120,782
                         *  triplets. We call this graph main facts.   Then we held out 8,331,147
                         *  entities from main facts and regard them as out-of-kb entities.  Under
                         *  such a setting, from  main  facts,  we  held  out  all  the  triplets
                         *  involving out-of-kb entities,  as well as 24,610,400 triplets that don’t
                         *  contain out-of-kb entities. Held-out triplets are used for validation
                         *  and testing; the remaining triplets are used for training.
                         */
                        double random = Math.random();
                        if (random < TEST_HOLDOUT) {
                            testWriter.write(
                                    String.format("%s\t%s\t%s\n", headId, tailId, relation));
                        } else if (random < TEST_HOLDOUT + VALID_HOLDOUT) {
                            validWriter.write(
                                    String.format("%s\t%s\t%s\n", headId, tailId, relation));
                        } else {
                            trainWriter.write(
                                    String.format("%s\t%s\t%s\n", headId, tailId, relation));
                        }
                    }
                }
            }
        }

        Log.info("KGCompletion.createKb2eInputFiles", "Writing entity2id.txt...");
        try (BufferedWriter entityWriter =
                FileTools.bufferedWriter(ENTITY2ID_FILE)) {
            for (int i = 0; i < entityIndex.size(); i++) {
                entityWriter.write(String.format("%s\t%d\n", entityIndex.get(i), i));
            }
        }

        Log.info("KGCompletion.createKb2eInputFiles",
                "Writing relation2id.txt...");
        try (BufferedWriter relationWriter =
                FileTools.bufferedWriter(RELATION2ID_FILE)) {
            for (int i = 0; i < KGRelation.values().length; i++) {
                String relation = KGRelation.values()[i].name();
                relationWriter.write(String.format("%s\t%d\n", relation, i));
            }
        }

        Log.info("KGCompletion.createKb2eInputFiles", "Writing e1_e2.txt...");
        try (BufferedWriter pairWriter =
                FileTools.bufferedWriter(E1_E2_FILE)) {
            // TODO
        }
    }

    /**
     * Returns a mapping from entityId to its adjacency lists based on the relations in KGRelation.
     */
    private static Map<String, List<Set<String>>> getAdjacencyMap() {
        Map<String, List<Set<String>>> relationMap = new ConcurrentHashMap<>();
        try (DiffbotEntityReader der = new DiffbotEntityReader(ENTITY_DUMP.getPath())) {
            int batchCount = 0;
            for (List<DiffbotEntity> batch = der.nextBatch(10000); batch != null;
                 batch = der.nextBatch(10000)){
                batchCount++;
                batch.parallelStream().forEach(e -> {
                    List<Set<String>> adjacencyLists = KGRelation.getRelations(e);
                    int relationCount = adjacencyLists.stream().mapToInt(Set::size).sum();
                    if (relationCount >= MINIMUM_RELATIONS) {
                        relationMap.put(e.id, adjacencyLists);
                    }
                });
                Log.info("KGCompletion.getAdjacencyMap",
                        "Read " + batchCount + " batches");
            }
        }

        return relationMap;
    }

    public static void main(String[] args) throws Exception {
        createKb2eInputFiles();
    }
}

