package com.diffbot.ml;

import com.diffbot.entities.DiffbotEntity;
import com.diffbot.kg.knowledgefusion.SecondPassKnowledgeFusion;
import com.diffbot.toolbox.FileTools;
import com.diffbot.utils.DiffbotEntityReader;
import com.esotericsoftware.minlog.Log;
import com.google.common.base.Splitter;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class KGCompletion {
    private static final Splitter RELATION_SPLITTER = Splitter.on("\t");
    private static final File ENTITY_DUMP = new File(
            "/home/ubuntu/test/data/bleeding_edge/common/knowledgeFusionSecondPass",
            SecondPassKnowledgeFusion.FINAL_OUTPUT_PATH);
    private static final int MINIMUM_RELATIONS = 10;

    static final File KB2E_DIRECTORY = new File("data/kb2e");
    static final File TRAIN_FILE = new File(KB2E_DIRECTORY, "train.txt");
    static final File TEST_FILE = new File(KB2E_DIRECTORY, "test.txt");
    static final File ENTITY2ID_FILE = new File(KB2E_DIRECTORY, "entity2id.txt");
    static final File RELATION2ID_FILE = new File(KB2E_DIRECTORY, "relation2id.txt");

    /**
     * Creates the inputs for com.diffbot.ml.TransE and other embeddings in
     * https://github.com/thunlp/KB2E
     *
     * Creates the following files in KB2E_DIRECTORY:
     * train.txt: training file, format (e1, e2, rel).
     * test.txt: test file, same format as train.txt.
     * entity2id.txt: all entities and corresponding ids, one per line.
     * relation2id.txt: all relations and corresponding ids, one per line.
     * e1_e2.txt: all top-500 entity pairs mentioned in the task entity prediction.
     */
    private static void createKb2eInputFiles() throws IOException {
        final float TEST_HOLDOUT = 0;
        long start = System.currentTimeMillis();

        KB2E_DIRECTORY.mkdirs();

        Set<String> entityIds = ConcurrentHashMap.newKeySet();
        try (DiffbotEntityReader der = new DiffbotEntityReader(ENTITY_DUMP.getPath());
                BufferedWriter trainWriter = FileTools.bufferedWriter(TRAIN_FILE);
                BufferedWriter testWriter = FileTools.bufferedWriter(TEST_FILE)) {
            int batchCount = 0;
            for (List<DiffbotEntity> batch = der.nextBatch(10000); batch != null;
                 batch = der.nextBatch(10000)){
                batchCount++;
                batch.parallelStream().forEach(e -> {
                    Set<String> adjacencyList = KGRelation.getRelations(e);
                    if (adjacencyList.size() < MINIMUM_RELATIONS) {
                        return;
                    }

                    String headId = e.id;
                    entityIds.add(headId);
                    for (String relationPair : adjacencyList) {
                        Iterator<String> ids = RELATION_SPLITTER.split(relationPair).iterator();
                        int relationId = Integer.valueOf(ids.next());
                        String relation = KGRelation.values()[relationId].name();

                        String tailId = ids.next();
                        entityIds.add(tailId);

                        double random = Math.random();
                        try {
                            if (random < TEST_HOLDOUT) {
                                synchronized (testWriter) {
                                    testWriter.write(headId + "\t" + tailId + "\t" + relation + "\n");
                                }
                            } else {
                                synchronized (trainWriter) {
                                    trainWriter.write(headId + "\t" + tailId + "\t" + relation + "\n");
                                }
                            }
                        } catch (IOException ioe) {
                            Log.warn("KGCompletion.createKb2eInputFiles", ioe);
                        }
                    }
                });
                Log.info("KGCompletion.createKb2eInputFiles",
                        "Read " + batchCount + " batches with " + entityIds.size() +
                                " entities in " + (System.currentTimeMillis() - start) + "ms...");
            }
        }

        Log.info("KGCompletion.createKb2eInputFiles", "Writing entity2id.txt...");
        try (BufferedWriter entityWriter = FileTools.bufferedWriter(ENTITY2ID_FILE)) {
            int i = 0;
            for (String entityId : entityIds) {
                entityWriter.write(entityId + "\t" + i + "\n");
                i++;
            }
        }

        Log.info("KGCompletion.createKb2eInputFiles",
                "Writing relation2id.txt...");
        try (BufferedWriter relationWriter = FileTools.bufferedWriter(RELATION2ID_FILE)) {
            for (int i = 0; i < KGRelation.values().length; i++) {
                String relation = KGRelation.values()[i].name();
                relationWriter.write(relation + "\t" + i + "\n");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        createKb2eInputFiles();
    }
}

