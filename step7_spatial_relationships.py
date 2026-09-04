# ============================================
# Florent Poux Tutorial
# Step 7 - Spatial Relationship Computation
#             + Topology
# ============================================

import json
import numpy as np


# ============================================
# STEP 1: Load objects
# ============================================

def load_objects(file_path):

    with open(file_path, "r") as f:
        objects = json.load(f)

    print()
    print("=" * 60)
    print("STEP 7 - SPATIAL RELATIONSHIPS + TOPOLOGY")
    print("=" * 60)

    print()
    print(f"Loaded {len(objects)} objects")

    return objects


# ============================================
# STAGE 1: Topology Analytics
# ============================================

def is_contained(bounds1, bounds2):
    """Check if object1 is contained within object2."""

    return (
        np.all(
            np.array(bounds1["min"])
            >= np.array(bounds2["min"])
        )
        and
        np.all(
            np.array(bounds1["max"])
            <= np.array(bounds2["max"])
        )
    )


def are_adjacent(
    bounds1,
    bounds2,
    tolerance=0.1
):
    """Check whether two objects are adjacent."""

    for axis in range(3):

        if (
            abs(
                bounds1["max"][axis]
                - bounds2["min"][axis]
            ) < tolerance
            or
            abs(
                bounds2["max"][axis]
                - bounds1["min"][axis]
            ) < tolerance
        ):

            return True

    return False


# ============================================
# STAGE 2: Relationship Classification
# ============================================

def determine_relationship_type(
    obj1,
    obj2,
    threshold
):

    centroid1 = np.array(
        obj1["centroid"]
    )

    centroid2 = np.array(
        obj2["centroid"]
    )

    # ----------------------------------------
    # Distance
    # ----------------------------------------

    distance = np.linalg.norm(
        centroid1 - centroid2
    )

    if distance > threshold:

        return None

    # ----------------------------------------
    # Vertical relationship
    # ----------------------------------------

    z_diff = (
        centroid1[2]
        - centroid2[2]
    )

    if abs(z_diff) > 0.5:

        if z_diff > 0:
            return "above"
        else:
            return "below"

    # ----------------------------------------
    # Containment
    # ----------------------------------------

    bounds1 = obj1["bounds"]
    bounds2 = obj2["bounds"]

    if is_contained(
        bounds1,
        bounds2
    ):

        return "inside"

    elif is_contained(
        bounds2,
        bounds1
    ):

        return "contains"

    # ----------------------------------------
    # Adjacency
    # ----------------------------------------

    if are_adjacent(
        bounds1,
        bounds2,
        tolerance=0.3
    ):

        return "adjacent"

    # ----------------------------------------
    # Default
    # ----------------------------------------

    return "near"


# ============================================
# STAGE 3: Exhaustive Pairwise Analysis
# ============================================

def compute_spatial_relationships(
    objects,
    distance_threshold=2.0
):

    relationships = []

    object_names = list(
        objects.keys()
    )

    for i, obj1 in enumerate(
        object_names
    ):

        for j, obj2 in enumerate(
            object_names[i + 1:],
            i + 1
        ):

            rel_type = (
                determine_relationship_type(
                    objects[obj1],
                    objects[obj2],
                    distance_threshold
                )
            )

            if rel_type:

                relationships.append(
                    (
                        obj1,
                        obj2,
                        rel_type
                    )
                )

    return relationships


# ============================================
# STEP 4: Print relationships
# ============================================

def print_relationships(
    relationships
):

    print()
    print("=" * 60)
    print("SPATIAL RELATIONSHIPS")
    print("=" * 60)

    for obj1, obj2, relation in relationships:

        print(
            f"{obj1:20s}"
            f" -- {relation:10s} --> "
            f"{obj2}"
        )

    print()
    print(
        f"Total relationships: "
        f"{len(relationships)}"
    )


# ============================================
# STEP 5: Save relationships
# ============================================

def save_relationships(
    relationships,
    file_path
):

    data = []

    for obj1, obj2, relation in relationships:

        data.append({

            "object1": obj1,

            "object2": obj2,

            "relationship": relation
        })

    with open(
        file_path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

    print()
    print(
        f"Saved relationships to: "
        f"{file_path}"
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    # ----------------------------------------
    # Load objects
    # ----------------------------------------

    objects = load_objects(
        "output/objects.json"
    )

    # ----------------------------------------
    # Compute relationships
    # ----------------------------------------

    relationships = (
        compute_spatial_relationships(
            objects,
            distance_threshold=2.0
        )
    )

    # ----------------------------------------
    # Print
    # ----------------------------------------

    print_relationships(
        relationships
    )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    save_relationships(
        relationships,
        "output/spatial_relationships.json"
    )