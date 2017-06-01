package com.diffbot.ml;

import com.diffbot.entities.*;

import java.util.*;

public enum KGRelation {
    EMPLOYER {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof Person) {
                Person p = (Person) de;
                if (p.employments != null) {
                    p.employments.stream().filter(e -> e.employer != null)
                            .map(e -> e.employer.value.parseDiffbotIdFromUri())
                            .filter(Objects::nonNull).forEach(adjacentIds::add);
                }
            }
            return adjacentIds;
        }
    },
    SCHOOL {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof Person) {
                Person p = (Person) de;
                if (p.educations != null) {
                    p.educations.stream().filter(e -> e.institution != null)
                            .map(e -> e.institution.value.parseDiffbotIdFromUri())
                            .filter(Objects::nonNull).forEach(adjacentIds::add);
                }
            }
            return adjacentIds;
        }
    },
    SKILL {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof Person) {
                Person p = (Person) de;
                if (p.skills != null) {
                    p.skills.stream()
                            .map(s -> s.value.parseDiffbotIdFromUri())
                            .filter(Objects::nonNull).forEach(adjacentIds::add);
                }
            }
            return adjacentIds;
        }
    },
    LOCATION {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof Person) {
                Person p = (Person) de;
                adjacentIds.addAll(getLocationIds(p.locations));
            } else if (de instanceof Organization) {
                Organization o = (Organization) de;
                adjacentIds.addAll(getLocationIds(o.locations));
            }
            return adjacentIds;
        }
    },
    AREA_PART_OF {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof AdministrativeArea) {
                AdministrativeArea a = (AdministrativeArea) de;
                if (a.getIsPartOf() != null) {
                    a.getIsPartOf().stream()
                            .map(f -> f.value.parseDiffbotIdFromUri())
                            .filter(Objects::nonNull).forEach(adjacentIds::add);
                }
            }
            return adjacentIds;
        }
    };

    public abstract Set<String> getRelatedEntityIds(DiffbotEntity de);

    public static List<Set<String>> getRelations(DiffbotEntity de) {
        List<Set<String>> relatedEntityIds = new ArrayList<>();
        for (KGRelation r : KGRelation.values()) {
            relatedEntityIds.add(r.getRelatedEntityIds(de));
        }
        return relatedEntityIds;
    }

    private static Set<String> getLocationIds(List<Fact<Location>> locationFacts) {
        if (locationFacts == null) {
            return Collections.emptySet();
        }

        Set<String> locationIds = new HashSet<>();
        for (Fact<Location> locationFact : locationFacts) {
            Location location = locationFact.value;
            if (location.getCity() != null) {
                locationIds.add(location.getCity().parseDiffbotIdFromUri());
            }
            if (location.getSubregion() != null) {
                locationIds.add(location.getSubregion().parseDiffbotIdFromUri());
            }
            if (location.getRegion() != null) {
                locationIds.add(location.getRegion().parseDiffbotIdFromUri());
            }
            if (location.getCountry() != null) {
                locationIds.add(location.getCountry().parseDiffbotIdFromUri());
            }
        }
        locationIds.removeIf(Objects::isNull);

        return locationIds;
    }
}
