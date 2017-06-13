package com.diffbot.ml;

import com.diffbot.entities.*;

import java.util.*;

// TODO: enforce type-safety
public enum KGRelation {
    // TODO: add Gender
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
    EMPLOYMENT_CATEGORY {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof Person) {
                Person p = (Person) de;
                if (p.employments != null) {
                    p.employments.stream().filter(e -> e.categories != null)
                            .flatMap(e -> e.categories.stream())
                            .map(c -> c.value.parseDiffbotIdFromUri())
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
    PARENT_ORGANIZATION {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof Organization) {
                Organization o = (Organization) de;
                if (o.parentCompany != null) {
                    String parentId = o.parentCompany.value.parseDiffbotIdFromUri();
                    if (parentId != null) {
                        adjacentIds.add(parentId);
                    }
                }
            }
            return adjacentIds;
        }
    },
    SUBSIDIARY_ORGANIZATION {
        @Override public Set<String> getRelatedEntityIds(DiffbotEntity de) {
            Set<String> adjacentIds = new HashSet<>();
            if (de instanceof Organization) {
                Organization o = (Organization) de;
                if (o.subsidiaries != null) {
                    o.subsidiaries.stream()
                            .map(f -> f.value.parseDiffbotIdFromUri())
                            .filter(Objects::nonNull).forEach(adjacentIds::add);
                }
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

    public static Set<String> getRelations(DiffbotEntity de) {
        Set<String> relatedEntityIds = new HashSet<>();
        for (KGRelation r : KGRelation.values()) {
            r.getRelatedEntityIds(de).stream().map(id -> "" + r.ordinal() + "\t" + id)
                    .forEach(relatedEntityIds::add);
        }
        return relatedEntityIds;
    }

    // TODO: populate partOf fields using LocationElector.populateParents
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
