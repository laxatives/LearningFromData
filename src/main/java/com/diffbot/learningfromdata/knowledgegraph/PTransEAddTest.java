package com.diffbot.ml;

import com.diffbot.entities.Skill;
import com.diffbot.entities.enums.RoleCategory;
import com.diffbot.toolbox.FileTools;
import com.diffbot.utils.Pair;
import com.esotericsoftware.minlog.Log;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Inputs (provided by PTransEAddTrain):
 *     entity2id.txt
 *     relation2id.txt
 *     train.txt
 *     path2.txt
 *     test.txt
 *     entity2vec.txt
 *     relation2vec.txt
 *
 * TODO: refactor this into PTransEAddTrain as PTransEAdd
 * TODO: port to Spark/Tensorflow
 */
public class PTransEAddTest {
    private static final Splitter WHITESPACE_SPLITTER = Splitter.onPattern("\\s+").trimResults().omitEmptyStrings();
    private static final int RERANK_NUM = 500;
    private int entityCount = 0;
    private int relationCount = 0;

    private Map<String, Integer> entityToId = new HashMap<>();
    private Map<String, Integer> relationToId = new HashMap<>();
    private Map<Integer, String> idToEntity = new HashMap<>();
    private Map<Integer, String> idToRelation = new HashMap<>();

    private Map<Pair<String, Integer>, Float> pathConfidence = new HashMap<>();
    private Map<Pair<Integer, Integer>, List<Pair<int[], Float>>> pathResources = new HashMap<>();

    /**
     * A map of Pair<headId, relationId> -> Set<tailId> where every headId, readId, tailId triple
     * implies a positive label.
     */
    private Set<String> trainingTriples = new HashSet<>();

    private float[][] entityVec;
    private float[][] relationVec;

    public void test() throws IOException {
        prepare();
        run();
    }

    private void addRelations(int headId, int tailId, List<Pair<int[], Float>> pathResources) {
        if (!pathResources.isEmpty()) {
            this.pathResources.put(new Pair<>(headId, tailId), pathResources);
        }
    }

    private void prepare() throws IOException {
        Log.info("PTransEAddTest.prepare", "Loading entities from " +
                KGCompletion.ENTITY2ID_FILE + "...");
        try (BufferedReader br = FileTools.bufferedReader(KGCompletion.ENTITY2ID_FILE)) {
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                List<String> split = WHITESPACE_SPLITTER.splitToList(line);
                String entity = split.get(0);
                int id = Integer.valueOf(split.get(1));
                entityToId.put(entity, id);
                idToEntity.put(id, entity);
                entityCount++;
            }
        }

        Log.info("PTransEAddTest.prepare", "Loading relations from " +
                KGCompletion.RELATION2ID_FILE + "...");
        try (BufferedReader br = FileTools.bufferedReader(KGCompletion.RELATION2ID_FILE)) {
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                List<String> split = WHITESPACE_SPLITTER.splitToList(line);
                String relation = split.get(0);
                int id = Integer.valueOf(split.get(1));
                relationToId.put(relation, id);
                idToRelation.put(id, relation);
                // one for forward, one for inverse
                relationCount += 2;
            }
        }

        for (File pathFile : ImmutableList.of(Pcra.Mode.TEST.getPathResourceFile(), Pcra.PATH2_FILE)) {
            Log.info("PTransEAddTest.prepare", "Loading triples from " +
                    pathFile + "...");
            try (BufferedReader praReader = FileTools.bufferedReader(pathFile)) {
                for (String line = praReader.readLine(); line != null; line = praReader.readLine()) {
                    Iterator<String> triple = WHITESPACE_SPLITTER.split(line).iterator();
                    String head = triple.next();
                    String tail = triple.next();

                    if (!entityToId.containsKey(head)) {
                        Log.warn("PTransEAddTest.prepare", "Missing entity " + head);
                        continue;
                    }
                    if (!entityToId.containsKey(tail)) {
                        Log.warn("PTransEAddTest.prepare", "Missing entity " + tail);
                        continue;
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
                            path[j] = Integer.valueOf(pathParts.next());
                        }
                        if (pathParts.hasNext()) {
                            Log.error("PTransEAddTest.prepare", "Unexpected path2: " + pathLine);
                            return;
                        }

                        float resourceAllocation = Float.valueOf(parts.next());

                        pathResources.add(new Pair<>(path, resourceAllocation));
                    }

                    if (parts.hasNext()) {
                        Log.error("PTransEAddTest.prepare", "Unexpected pathResources: " + pathLine);
                        return;
                    }

                    addRelations(headId, tailId, pathResources);
                }
            }
        }

        Log.info("PTransEAddTest.prepare", "Loading triples from " +
                Pcra.Mode.TRAIN.getTriplesFile() + "...");
        try (BufferedReader trainReader = FileTools.bufferedReader(Pcra.Mode.TRAIN.getTriplesFile())) {
            for (String line = trainReader.readLine(); line != null; line = trainReader.readLine()) {
                Iterator<String> triple = WHITESPACE_SPLITTER.split(line).iterator();
                String head = triple.next();
                String tail = triple.next();
                String relation = triple.next();

                if (!entityToId.containsKey(head)) {
                    Log.warn("PTransEAddTest.prepare", "Missing entity " + head);
                    continue;
                } else if (!entityToId.containsKey(tail)) {
                    Log.warn("PTransEAddTest.prepare", "Missing entity " + tail);
                    continue;
                } else if (!relationToId.containsKey(relation)) {
                    Log.warn("PTransEAddTest.prepare", "Missing relation " + relation);
                    continue;
                }

                int headId = entityToId.get(head);
                int tailId = entityToId.get(tail);
                int relationId = relationToId.get(relation);
                trainingTriples.add(headId + "-" + tailId + "-" + relationId);
            }
        }

        Log.info("PTransEAddTest.prepare", "Loading path confidences from " +
                Pcra.CONFIDENCE_FILE + "...");
        try (BufferedReader confidenceReader = FileTools.bufferedReader(Pcra.CONFIDENCE_FILE)) {
            for (String line = confidenceReader.readLine(); line != null; line = confidenceReader.readLine()) {
                Iterator<String> parts = WHITESPACE_SPLITTER.split(line).iterator();
                int size = Integer.valueOf(parts.next());

                List<String> pathList = new ArrayList<>();
                Iterator<String> pathParts = Pcra.PATH_SPLITTER.split(parts.next()).iterator();
                for (int j = 0; j < size; j++) {
                    pathList.add(pathParts.next());
                }
                String path = pathList.stream().collect(Collectors.joining(" "));
                if (pathParts.hasNext()) {
                    Log.error("PTransEAddTest.prepare", "Unexpected confidence: " + line);
                    return;
                }

                if (parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected confidence path: " + line);
                    return;
                }

                String confidences = confidenceReader.readLine();

                Iterator<String> confidenceParts = WHITESPACE_SPLITTER.split(confidences).iterator();
                int confidenceLength = Integer.valueOf(confidenceParts.next());
                for (int i = 0; i < confidenceLength; i++) {
                    Integer relation = Integer.valueOf(confidenceParts.next());

                    float confidence = Float.valueOf(confidenceParts.next());
                    pathConfidence.put(new Pair<>(path, relation), confidence);

                    Log.debug(path + "," + relation + "->" + confidence);
                }

                if (parts.hasNext()) {
                    Log.error("PTransEAddTrain.prepare", "Unexpected confidences: " + confidences);
                    return;
                }
            }
        }
    }

    private void run() throws IOException {
        Log.info("PTransETest.run", "relationCount=" + relationCount +
                ", entityCount=" + entityCount);

        Log.info("Loading Entity Vector from " + PTransEAddTrain.ENTITY2VEC_FILE + "...");
        entityVec = new float[entityCount][PTransEAddTrain.N];
        try (BufferedReader entityVecReader = FileTools.bufferedReader(PTransEAddTrain.ENTITY2VEC_FILE)) {
            for (int i = 0; i < entityCount; i++) {
                String row = entityVecReader.readLine();
                Iterator<String> entries = WHITESPACE_SPLITTER.split(row).iterator();
                for (int j = 0; j < PTransEAddTrain.N; j++) {
                    entityVec[i][j] = Float.valueOf(entries.next());
                }

                if (entries.hasNext()) {
                    Log.error("PTransEAddTest.run", "Entity vector length mismatch on row " + i);
                    return;
                }
            }

            if (entityVecReader.readLine() != null) {
                Log.error("PTransEAddTest.run", "Entity vector count mismatch");
            }
        }

        Log.info("Loading Relation Vector from " + PTransEAddTrain.RELATION2VEC_FILE + "...");
        relationVec = new float[relationCount][PTransEAddTrain.N];
        try (BufferedReader relationVecReader = FileTools.bufferedReader(PTransEAddTrain.RELATION2VEC_FILE)) {
            for (int i = 0; i < relationCount; i++) {
                String row = relationVecReader.readLine();
                Iterator<String> entries = WHITESPACE_SPLITTER.split(row).iterator();
                for (int j = 0; j < PTransEAddTrain.N; j++) {
                    relationVec[i][j] = Float.valueOf(entries.next());
                }

                if (entries.hasNext()) {
                    Log.error("PTransEAddTest.run", "Relation vector length mismatch");
                    return;
                }
            }

            if (relationVecReader.readLine() != null) {
                Log.error("PTransEAddTest.run", "Relation vector count mismatch");
            }
        }

        for (int headId = 0; headId < entityCount; headId++) {
            String headDiffbotId = idToEntity.get(headId);
            if (!headDiffbotId.startsWith("P")) {
                continue;
            }

            // Infer skills
            inferTail(headId, KGRelation.SKILL.ordinal());

            // Infer employment category
            inferTail(headId, KGRelation.EMPLOYMENT_CATEGORY.ordinal());

            // Find similar
            getSimilar(headId);
        }
    }

    private void inferTail(int headId, int relationId) {
        List<Pair<Integer, Double>> candidateScores = new ArrayList<>();

        // TODO: enforce entity type generically
        String tailTypePrefix;
        double minScore;
        if (KGRelation.SKILL.ordinal() == relationId) {
            tailTypePrefix = "S";
            minScore = 90;
        } else if (KGRelation.EMPLOYMENT_CATEGORY.ordinal() == relationId) {
            tailTypePrefix = "RC";
            minScore = 88;
        } else {
            throw new UnsupportedOperationException(
                    "PTransEAddTest.inferTail does not support relationId " + relationId);
        }

        for (int i = 0; i < entityCount; i++) {
            String tailDiffbotId = idToEntity.get(i);
            if (!tailDiffbotId.startsWith(tailTypePrefix)) {
                continue;
            }

            double score = scoreTriple(headId, i, relationId, false);
            if (score > minScore) {
                candidateScores.add(new Pair<>(i, score));
            }
        }
        candidateScores.sort(Comparator.comparing(s -> s.second));
        for (int i = candidateScores.size() - 1; i >= Math.max(candidateScores.size() - RERANK_NUM, 0); i--) {
            candidateScores.get(i).second = scoreTriple(headId, candidateScores.get(i).first,
                    relationId, true);
        }
        candidateScores.sort(Comparator.comparing(s -> s.second));

        List<String> results = Lists.newArrayList("Best guesses given http://localhost:9200/diffbot_entity/Person/" +
                idToEntity.get(headId) + " and relation " + idToRelation.get(relationId) + ":");
        boolean outOfSampleExists = false;
        for (int i = candidateScores.size() - 1; i >= 0; i--) {
            boolean inSample = trainingTriples.contains(headId + "-" +
                    candidateScores.get(i).first + "-" + relationId);
            if (candidateScores.size() - i <= 8) {
                outOfSampleExists |= !inSample;
                results.add("\t" + getDiffbotName(candidateScores.get(i).first, relationId) + "\tscored\t" +
                        candidateScores.get(i).second + (inSample ? "\t(in sample)" : ""));
            }
        }
        if (outOfSampleExists && results.size() > 1) {
            Log.info("PTransEAddTest.inferTail", results.stream()
                    .collect(Collectors.joining("\n")));
        }
    }

    // TODO: enforce entity type (other than Person)
    private void getSimilar(int headId) {
        List<Pair<Integer, Double>> candidateScores = new ArrayList<>();
        for (int i = 0; i < entityCount; i++) {
            String tailDiffbotId = idToEntity.get(i);
            if (headId == i || !tailDiffbotId.startsWith("P")) {
                continue;
            }

            double similarity = scoreSimilarity(headId, i);
            if (similarity > 95) {
                candidateScores.add(new Pair<>(i, similarity));
            }
        }
        candidateScores.sort(Comparator.comparing(s -> s.second));

        List<String> results = Lists.newArrayList("Similar entities to http://localhost:9200/diffbot_entity/Person/" +
                idToEntity.get(headId));
        for (int i = candidateScores.size() - 1; i >= 0; i--) {
            if (candidateScores.size() - i <= 4) {
                results.add("\thttp://localhost:9200/diffbot_entity/Person/" +
                        idToEntity.get(candidateScores.get(i).first) + "\tscored\t" +
                        candidateScores.get(i).second);
            }
        }
        if (results.size() > 1) {
            Log.info("PTransEAddTest", results.stream()
                    .collect(Collectors.joining("\n")));
        }
    }

    private String getDiffbotName(int entityId, int relationId) {
        if (KGRelation.SKILL.ordinal() == relationId) {
            Skill skill = Skill.findBySkillId(idToEntity.get(entityId).replaceFirst("S_", ""));
            if (skill != null) {
                return skill.name.value;
            }
        } else if (KGRelation.EMPLOYMENT_CATEGORY.ordinal() == relationId) {
            String rcName = RoleCategory.codeToName(idToEntity.get(entityId));
            if (rcName != null) {
                return rcName;
            }
        }
        return idToEntity.get(entityId);
    }

    private double scoreSimilarity(int e1, int e2) {
        double sum = 0;
        for (int j = 0; j < PTransEAddTrain.N; j++) {
//            sum -= 10 * Math.abs(entityVec[e1][j] - entityVec[e2][j]);
            sum += entityVec[e1][j] * entityVec[e2][j];
        }

        double norm1 = PTransEAddTrain.l2Norm(entityVec[e1]);
        if (norm1 > 0) {
            sum /= norm1;
        }

        double norm2 = PTransEAddTrain.l2Norm(entityVec[e2]);
        if (norm2 > 0) {
            sum /= norm2;
        }

        return 100 * sum;
    }

    private double scoreTriple(int e1, int e2, int rel, boolean pathBoost) {
        double sum = 100;

        int inverseRelationId = rel + (relationCount / 2);

        for (int j = 0; j < PTransEAddTrain.N; j++) {
            sum -= Math.abs(entityVec[e2][j] - entityVec[e1][j] - relationVec[rel][j]);
            sum -= Math.abs(entityVec[e1][j] - entityVec[e2][j] - relationVec[inverseRelationId][j]);
        }

        if (pathBoost) {
            List<Pair<int[], Float>> path_list =
                    pathResources.getOrDefault(new Pair<>(e1, e2), Collections.emptyList());
            for (Pair<int[], Float> path : path_list) {
                int[] rel_path = path.first;
                String pathString = Arrays.stream(rel_path)
                        .boxed()
                        .map(String::valueOf)
                        .collect(Collectors.joining(" "));
                double pr = path.second;
                double pr_path = pathConfidence.getOrDefault(
                        new Pair<>(pathString, rel), 0f);

                sum -= scorePath(rel, rel_path) * pr * pr_path;
            }

            // TODO: this loop doesn't appear to do anything (including in the reference version)
            List<Pair<int[], Float>> reverse_path_list =
                    pathResources.getOrDefault(new Pair<>(e2, e1), Collections.emptyList());
            for (Pair<int[], Float> path : reverse_path_list) {
                int[] rel_path = path.first;
                String pathString = Arrays.stream(rel_path)
                        .boxed()
                        .map(String::valueOf)
                        .collect(Collectors.joining(" "));
                double pr = path.second;
                double pr_path = pathConfidence.getOrDefault(
                        new Pair<>(pathString, inverseRelationId), 0f);
                sum -= scorePath(inverseRelationId, rel_path) * pr * pr_path;
            }
        }

        //        System.out.println("" + e1 + "," + e2 + "," + rel +" -> " + sum);

        return sum;
    }

    private double scorePath(int r1, int[] rel_path) {
        double sum = 0;
        for (int k = 0; k < PTransEAddTrain.N; k++) {
            double tmp = relationVec[r1][k];
            for (int j : rel_path) {
                tmp -= relationVec[j][k];
            }

            sum -= Math.abs(tmp);
        }
        return sum;
    }

    public static void main(String[] args) throws Exception {
        PTransEAddTest add = new PTransEAddTest();
        add.test();
    }
}
