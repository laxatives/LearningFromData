package com.diffbot.ml;

import com.diffbot.toolbox.FileTools;
import com.diffbot.utils.Pair;
import com.esotericsoftware.minlog.Log;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.koloboke.collect.map.hash.*;
import com.koloboke.collect.set.hash.HashIntSet;
import com.koloboke.collect.set.hash.HashIntSets;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * TODO: refactor this and make it sane
 * TODO: refactor this into PTransEAddTrain as PTransEAdd
 * TODO: port to Spark/Tensorflow
 */
public class PTransEAddTest {
    private static final Splitter WHITESPACE_SPLITTER = Splitter.onPattern("\\s+").trimResults();
    private static final File N2N_FILE = new File(KGCompletion.KB2E_DIRECTORY, "n2n.txt");
    private static final int RERANK_NUM = 500;
    private int entityCount = 0;
    private int relationCount = 0;

    private HashObjIntMap<String> entityToId = HashObjIntMaps.newUpdatableMap();
    private HashObjIntMap<String> relationToId = HashObjIntMaps.newUpdatableMap();
    private HashObjFloatMap<Pair<String, Integer>> pathConfidence = HashObjFloatMaps.newMutableMap();
    private List<Integer> headIds = new ArrayList<>(); // list of headIds
    private List<Integer> tailIds = new ArrayList<>(); // list of tailIds
    private List<Integer> relationIds = new ArrayList<>(); // list of relationIds
    private HashObjObjMap<Pair<Integer, Integer>, List<Pair<int[], Float>>> fb_path =
            HashObjObjMaps.newMutableMap();
    private HashIntObjMap<HashIntIntMap> entity2num = HashIntObjMaps.newMutableMap();
    private HashIntIntMap e2num = HashIntIntMaps.newMutableMap();
    private List<Integer> rel_type = new ArrayList<>(); // list of relationIds

    /**
     * A map of Pair<headId, relationId> -> Set<tailId> where every headId, readId, tailId triple
     * implies a positive label.
     */
    private HashObjObjMap<Pair<Integer, Integer>, HashIntSet> positiveTriples =
            HashObjObjMaps.newUpdatableMap();

    private float[][] entityVec;
    private float[][] relationVec;
    private boolean used = true;

    public void test() throws IOException {
        prepare();
        run();
    }

    private void addRelations(int headId, int tailId, int relationId, boolean addIds) {
        if (addIds) {
            headIds.add(headId);
            tailIds.add(tailId);
            relationIds.add(relationId);
        }

        Pair<Integer, Integer> key = new Pair<>(headId, relationId);
        positiveTriples.putIfAbsent(key, HashIntSets.newMutableSet());
        positiveTriples.get(key).add(tailId);
    }

    private void addRelations(int headId, int tailId, int relationId,
            List<Pair<int[], Float>> pathResources) {
        addRelations(headId, tailId, relationId, true);
        if (!pathResources.isEmpty()) {
            fb_path.put(new Pair<>(headId, tailId), pathResources);
        }
    }

    private void prepare() throws IOException {
        try (BufferedReader br = FileTools.bufferedReader(KGCompletion.ENTITY2ID_FILE)) {
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                List<String> split = WHITESPACE_SPLITTER.splitToList(line);
                String entity = split.get(0);
                int id = Integer.valueOf(split.get(1));
                entityToId.put(entity, id);
                entityCount++;
            }
        }

        try (BufferedReader br = FileTools.bufferedReader(KGCompletion.RELATION2ID_FILE)) {
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                List<String> split = WHITESPACE_SPLITTER.splitToList(line);
                String relation = split.get(0);
                int id = Integer.valueOf(split.get(1));
                relationToId.put(relation, id);
                // one for forward, one for inverse
                relationCount += 2;
            }
        }

        for (File pathFile : ImmutableList.of(Pcra.Mode.TEST.getPraFile(), Pcra.PATH2_FILE)) {
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
                        Log.error("PTransEAddTest.prepare", "Unexpected pathResources: " + pathResources);
                        return;
                    }

                    if (pathFile.equals(Pcra.Mode.TEST.getPraFile())) {
                        int relationId = Integer.valueOf(triple.next());
                        addRelations(headId, tailId, relationId, b);
                    } else {
                        addRelations(headId, tailId, -1, b);
                    }
                }
            }
        }

        try (BufferedReader praReader = FileTools.bufferedReader(Pcra.Mode.TRAIN.getTriplesFile())) {
            for (String line = praReader.readLine(); line != null; line = praReader.readLine()) {
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

                int headId = entityToId.getInt(head);
                int tailId = entityToId.getInt(tail);
                int relationId = relationToId.getInt(relation);

                entity2num.putIfAbsent(relationId, HashIntIntMaps.newMutableMap());
                entity2num.get(relationId).addValue(headId, 1, 0);
                entity2num.putIfAbsent(relationId, HashIntIntMaps.newMutableMap());
                entity2num.get(relationId).addValue(tailId, 1, 0);
                e2num.addValue(headId, 1, 0);
                e2num.addValue(tailId, 1, 0);
                addRelations(headId, tailId, relationId, false);
            }
        }

        try (BufferedReader validReader = FileTools.bufferedReader(KGCompletion.VALID_FILE)) {
            for (String line = validReader.readLine(); line != null; line = validReader.readLine()) {
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

                int headId = entityToId.getInt(head);
                int tailId = entityToId.getInt(tail);
                int relationId = relationToId.getInt(relation);
                addRelations(headId, tailId, relationId, false);
            }
        }

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

        // TODO: what is this file...?
        try (BufferedReader n2nReader = FileTools.bufferedReader(N2N_FILE)) {
            for (String line = n2nReader.readLine(); line != null; line = n2nReader.readLine()) {
                Iterator<String> parts = WHITESPACE_SPLITTER.split(line).iterator();
                float x = Float.valueOf(parts.next());
                float y = Float.valueOf(parts.next());

                // TODO: I have no idea what this does
                if (x < 1.5) {
                    if (y < 1.5) {
                        rel_type.add(0);
                    } else {
                        rel_type.add(1);
                    }
                } else {
                    if (y < 1.5) {
                        rel_type.add(2);
                    } else {
                        rel_type.add(3);
                    }
                }
            }
        }
    }

    private void run() throws IOException {
        Log.info("PTransETest.run", "relationCount=" + relationCount +
                ", entityCount=" + entityCount);

        entityVec = new float[entityCount][PTransEAddTrain.N];
        try (BufferedReader entityVecReader = FileTools.bufferedReader(PTransEAddTrain.ENTITY2VEC_FILE)) {
            for (int i = 0; i < entityCount; i++) {
                String row = entityVecReader.readLine();
                Iterator<String> entries = WHITESPACE_SPLITTER.split(row).iterator();
                for (int j = 0; j < PTransEAddTrain.N; j++) {
                    entityVec[i][j] = Float.valueOf(entries.next());
                }

                if (entries.hasNext()) {
                    Log.error("PTransEAddTest.run", "Entity vector length mismatch");
                    return;
                }
            }

            if (entityVecReader.readLine() != null) {
                Log.error("PTransEAddTest.run", "Entity vector count mismatch");
            }
        }

        relationVec = new float[relationCount][PTransEAddTrain.N];
        try (BufferedReader relationVecReader = FileTools.bufferedReader(PTransEAddTrain.RELATION2VEC_FILE)) {
            for (int i = 0; i < entityCount; i++) {
                String row = relationVecReader.readLine();
                Iterator<String> entries = WHITESPACE_SPLITTER.split(row).iterator();
                for (int j = 0; j < PTransEAddTrain.N; j++) {
                    relationVec[i][j] = Float.valueOf(entries.next());
                }

                if (entries.hasNext()) {
                    Log.error("PTransEAddTest.run", "Relation vector length mismatch");
                    return;
                } else if (PTransEAddTrain.vectorLength(relationVec[i]) > 1) {
                    Log.error("PTransEAddTest.run", "Vector should be normalized: " + row);
                }
            }

            if (relationVecReader.readLine() != null) {
                Log.error("PTransEAddTest.run", "Relation vector count mismatch");
            }
        }

        double lsum = 0;
        double lsum_filter = 0;
        double mid_sum = 0;
        double mid_sum_filter = 0;
        double rsum = 0;
        double rsum_filter = 0;

        double lp_n = 0;
        double lp_n_filter = 0;
        double mid_p_n = 0;
        double mid_p_n_filter = 0;
        double rp_n = 0;
        double rp_n_filter = 0;

        double l_one2one = 0;
        double r_one2one = 0;
        double one2one_num = 0;
        double l_n2one = 0;
        double r_n2one = 0;
        double n2one_num = 0;
        double l_one2n = 0;
        double r_one2n = 0;
        double one2n_num = 0;
        double l_n2n = 0;
        double r_n2n = 0;
        double n2n_num = 0;

        HashIntFloatMap lsum_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap lsum_filter_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap mid_sum_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap mid_sum_filter_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap rsum_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap rsum_filter_r = HashIntFloatMaps.newMutableMap();

        HashIntFloatMap lp_n_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap lp_n_filter_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap mid_p_n_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap mid_p_n_filter_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap rp_n_r = HashIntFloatMaps.newMutableMap();
        HashIntFloatMap rp_n_filter_r = HashIntFloatMaps.newMutableMap();

        HashIntIntMap rel_num = HashIntIntMaps.newMutableMap();

        int hit_n = 1;
        HashObjIntMap<Pair<Integer, Integer>> e1_e2 = HashObjIntMaps.newMutableMap();
        for (int testid = 0; testid < tailIds.size() / 2; testid++) {
            int h = headIds.get(testid * 2);
            // TODO: rename l -> tailId
            int l = tailIds.get(testid * 2);
            int rel = tailIds.get(testid * 2);

            rel_num.addValue(rel, 1, 0);
            List<Pair<Integer, Double>> a = new ArrayList<>();

            // TODO: these should be enum
            int relationType = rel_type.get(rel);
            if (relationType == 0) {
                one2one_num++;
            } else if (relationType == 1) {
                n2one_num++;
            } else if (relationType == 2) {
                one2n_num++;
            } else if (relationType == 3) {
                n2n_num++;
            }

            int filter = 0;

            // Head entities
            for (int i = 0; i < entityCount; i++) {
                a.add(new Pair<>(i, calc_sum(i, l, rel, false)));
            }

            Collections.sort(a, Comparator.comparing(s -> s.second));
            for (int i = a.size() - 1; i >= a.size() - RERANK_NUM; i--) {
                a.get(i).second = calc_sum(a.get(i).first, l, rel, true);
            }

            Collections.sort(a, Comparator.comparing(s -> s.second));
            for (int i = a.size() - 1; i >= 0; i--) {
                if (a.size() - i <= RERANK_NUM) {
                    e1_e2.put(new Pair<>(a.get(i).first, l), 1);
                }

                if (!positiveTriples.get(new Pair<>(a.get(i).first, rel)).contains(l)) {
                    filter++;
                }

                if (a.get(i).first == h) {
                    lsum += a.size() - i;
                    lsum_filter += filter + 1;
                    lsum_r.addValue(rel, a.size() - 1, 0);
                    lsum_filter_r.addValue(rel, filter + 1, 0);

                    if (a.size() - 1 <= hit_n) {
                        lp_n += 1;
                        lp_n_r.addValue(rel, 1, 0);
                    }

                    if (filter < hit_n) {
                        lp_n_filter++;
                        lp_n_filter_r.addValue(rel, 1, 0);
                        if (relationType == 0) {
                            l_one2one++;
                        } else if (relationType == 1) {
                            l_n2one++;
                        } else if (relationType == 2) {
                            l_one2n++;
                        } else if (relationType == 3) {
                            l_n2n++;
                        }
                    }
                }
            }

            // Tail entities
            a.clear();
            filter = 0;
            for (int i = 0; i < entityCount; i++) {
                a.add(new Pair<>(i, calc_sum(h, i, rel, false)));
            }

            Collections.sort(a, Comparator.comparing(s -> s.second));
            for (int i = a.size() - 1; i >= a.size() - RERANK_NUM; i--) {
                a.get(i).second = calc_sum(h, a.get(i).first, rel, true);
            }

            Collections.sort(a, Comparator.comparing(s -> s.second));
            for (int i = a.size() - 1; i >= 0; i--) {
                if (a.size() - i <= RERANK_NUM) {
                    e1_e2.put(new Pair<>(h, a.get(i).first), 1);
                }

                if (!positiveTriples.get(new Pair<>(h, rel)).contains(a.get(i).first)) {
                    filter++;
                }

                if (a.get(i).first == l) {
                    rsum += a.size() - i;
                    rsum_filter += filter + 1;
                    rsum_r.addValue(rel, a.size() - 1, 0);
                    rsum_filter_r.addValue(rel, filter + 1, 0);

                    if (a.size() - 1 <= hit_n) {
                        rp_n += 1;
                        rp_n_r.addValue(rel, 1, 0);
                    }

                    if (filter < hit_n) {
                        rp_n_filter++;
                        rp_n_filter_r.addValue(rel, 1, 0);
                        if (relationType == 0) {
                            r_one2one++;
                        } else if (relationType == 1) {
                            r_n2one++;
                        } else if (relationType == 2) {
                            r_one2n++;
                        } else if (relationType == 3) {
                            r_n2n++;
                        }
                    }
                }
            }

            // Relations
            a.clear();
            filter = 0;
            for (int i = 0; i < relationCount; i++) {
                a.add(new Pair<>(i, calc_sum(h, l, i, false)));
            }
            Collections.sort(a, Comparator.comparing(s -> s.second));
            for (int i = a.size() - 1; i >= 0; i--) {
                if (a.size() - i <= RERANK_NUM) {
                    e1_e2.put(new Pair<>(h, a.get(i).first), 1);
                }

                if (!positiveTriples.get(new Pair<>(h, a.get(i).first)).contains(l)) {
                    filter++;
                }

                if (a.get(i).first == rel) {
                    mid_sum += a.size() - i;
                    mid_sum_filter += filter + 1;
                    mid_sum_r.addValue(rel, a.size() - 1, 0);
                    mid_sum_filter_r.addValue(rel, filter + 1, 0);

                    if (a.size() - 1 <= hit_n) {
                        mid_p_n += 1;
                        mid_p_n_r.addValue(rel, 1, 0);
                    }

                    if (filter < hit_n) {
                        mid_p_n_filter++;
                        mid_p_n_filter_r.addValue(rel, 1, 0);
                    }
                }
            }

            if (testid % 100 == 0) {
                int i = testid + 1;
                Log.info(testid + ":\t" + (lsum/i) + " " + (lp_n/i) + " " + (rsum/i) + " " +
                        (rp_n/i) + "\t" + (lsum_filter/i) + " " + (lp_n_filter/i) + " " +
                        (rsum_filter/i) + " " + (rp_n_filter/i));
                Log.info("\t" + (mid_sum/i) + " " + (mid_p_n/i) + "\t" + (mid_sum_filter/i) +
                        " " + (mid_p_n_filter/i));
                // TODO: more logging
            }
        }
    }

    private double calc_sum(int e1, int e2, int rel, boolean used) {
        double sum = 0;
        if (PTransEAddTrain.L1_FLAG) {
            for (int j = 0; j < PTransEAddTrain.N; j++) {
                sum -= Math.abs(entityVec[e2][j] - entityVec[e1][j] - relationVec[rel][j]);
                sum -= Math.abs(entityVec[e1][j] - entityVec[e2][j] -
                        relationVec[rel + (relationCount/2)][j]);
            }
        } else {
            for (int j = 0; j < PTransEAddTrain.N; j++) {
                double delta = entityVec[e2][j] - entityVec[e1][j] - relationVec[rel][j];
                sum -= delta * delta;
                double invDelta = Math.abs(entityVec[e1][j] - entityVec[e2][j] -
                        relationVec[rel + (relationCount/2)][j]);
                sum -= invDelta * invDelta;
            }
        }

        int h = e1;
        int l = e2;

        if (used) {
            List<Pair<int[], Float>> path_list =
                    fb_path.getOrDefault(new Pair<>(h,l), Collections.emptyList());
            for (Pair<int[], Float> path : path_list) {
                int[] rel_path = path.first;
                String pathString = Arrays.stream(rel_path)
                        .boxed()
                        .map(String::valueOf)
                        .collect(Collectors.joining(" "));
                double pr = path.second;
                double pr_path = pathConfidence
                        .getOrDefault(new Pair<>(pathString, rel), 0);
                sum += calc_path(rel, rel_path) * pr * pr_path;
            }

            List<Pair<int[], Float>> reverse_path_list =
                    fb_path.getOrDefault(new Pair<>(l, h), Collections.emptyList());
            for (Pair<int[], Float> path : path_list) {
                int[] rel_path = path.first;
                String pathString = Arrays.stream(rel_path)
                        .boxed()
                        .map(String::valueOf)
                        .collect(Collectors.joining(" "));
                double pr = path.second;
                double pr_path = pathConfidence
                        .getOrDefault(new Pair<>(pathString, rel + (relationCount / 2)), 0);
                sum += calc_path(rel + (relationCount / 2), rel_path) * pr * pr_path;
            }
        }

        return sum;
    }

    double calc_path(int r1, int[] rel_path) {
        double sum = 0;
        for (int k = 0; k < PTransEAddTrain.N; k++) {
            double tmp = relationVec[r1][k];
            for (int j = 0; j < rel_path.length; j++) {
                tmp -= relationVec[rel_path[j]][k];
            }

            if (PTransEAddTrain.L1_FLAG) {
                sum += Math.abs(tmp);
            } else {
                sum += tmp * tmp;
            }
        }
        return sum;
    }

    public static void main(String[] args) throws Exception {
        PTransEAddTest add = new PTransEAddTest();
        add.test();
    }
}
