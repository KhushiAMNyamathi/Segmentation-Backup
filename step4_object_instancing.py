# ============================================
# Florent Poux Tutorial
# Step 4 - Semantic Point Cloud Object Instancing
# ============================================

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import json
import os
from scipy.spatial import ConvexHull


# ============================================
# STEP 1: Load semantic point cloud
# ============================================

def load_semantic_point_cloud(
    file_path,
    column_name="semantic_label"
):
    """Load semantic point cloud from CSV."""

    df = pd.read_csv(
        file_path,
        delimiter=";"
    )

    class_names = [
        "ceiling",
        "floor",
        "wall",
        "chair",
        "furniture",
        "table"
    ]

    label_map = {
        float(i): class_names[i]
        for i in range(len(class_names))
    }

    df[column_name] = df[column_name].map(label_map)

    # Same sampling approach as the tutorial
    sample_size = min(150000, len(df))

    return df.sample(
        n=sample_size,
        random_state=1
    )


# ============================================
# STEP 4: Extract semantic objects
# ============================================

def extract_semantic_objects(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 10
):
    """
    Extract individual objects from semantic
    point cloud using DBSCAN clustering.
    """

    objects = {}

    # ----------------------------------------
    # Process each semantic class separately
    # ----------------------------------------

    for label in df["semantic_label"].dropna().unique():

        print()
        print("=" * 50)
        print(f"Processing semantic class: {label}")
        print("=" * 50)

        # Get only points belonging to this class
        label_points = df[
            df["semantic_label"] == label
        ].copy()

        print(
            f"Points in class: "
            f"{len(label_points):,}"
        )

        # Not enough points to cluster
        if len(label_points) < min_samples:

            print(
                "Skipping - not enough points."
            )

            continue

        # ------------------------------------
        # Get XYZ coordinates
        # ------------------------------------

        coords = label_points[
            ["x", "y", "z"]
        ].values

        # ------------------------------------
        # DBSCAN
        # ------------------------------------

        print(
            f"Running DBSCAN "
            f"(eps={eps}, "
            f"min_samples={min_samples})..."
        )

        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit(coords)

        # Add cluster labels
        label_points["cluster"] = (
            clustering.labels_
        )

        cluster_ids = np.unique(
            clustering.labels_
        )

        # ------------------------------------
        # Print clustering information
        # ------------------------------------

        number_of_clusters = len(
            cluster_ids[cluster_ids != -1]
        )

        noise_count = np.sum(
            clustering.labels_ == -1
        )

        print(
            f"Objects found: "
            f"{number_of_clusters}"
        )

        print(
            f"Noise points: "
            f"{noise_count:,}"
        )

        # ------------------------------------
        # Create object for every cluster
        # ------------------------------------

        for cluster_id in cluster_ids:

            # -1 means DBSCAN noise
            if cluster_id == -1:
                continue

            cluster_points = (
                label_points[
                    label_points["cluster"]
                    == cluster_id
                ].copy()
            )

            # Object name
            object_key = (
                f"{label}_{cluster_id}"
            )

            # --------------------------------
            # Centroid
            # --------------------------------

            centroid = cluster_points[
                ["x", "y", "z"]
            ].mean().values

            # --------------------------------
            # Bounding box
            # --------------------------------

            bounds = {
                "min": cluster_points[
                    ["x", "y", "z"]
                ].min().values,

                "max": cluster_points[
                    ["x", "y", "z"]
                ].max().values
            }

            # --------------------------------
            # Store object
            # --------------------------------

            objects[object_key] = {

                "points": cluster_points,

                "centroid": centroid,

                "bounds": bounds,

                "semantic_label": label,

                "point_count": len(
                    cluster_points
                )
            }

    return objects

def estimate_surface_area(points):
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(points)
        return hull.area
    except:
        return 0.0

def compute_object_features(objects):
    """Compute geometric and semantic features for each object."""

    features = {}

    for obj_name, obj_data in objects.items():

        points = obj_data["points"][
            ["x", "y", "z"]
        ].values

        # Geometric features
        volume = np.prod(
            obj_data["bounds"]["max"]
            - obj_data["bounds"]["min"]
        )

        surface_area = estimate_surface_area(
            points
        )

        compactness = (
            (surface_area ** 3)
            / (
                36
                * np.pi
                * volume ** 2
            )
            if volume > 0
            else 0
        )

        features[obj_name] = {

            "volume": volume,

            "surface_area": surface_area,

            "compactness": compactness,

            "height":
                obj_data["bounds"]["max"][2]
                - obj_data["bounds"]["min"][2],

            "semantic_label":
                obj_data["semantic_label"],

            "centroid":
                obj_data["centroid"],

            "point_density":
                obj_data["point_count"]
                / volume
                if volume > 0
                else 0
        }

    return features
# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("FLORENT POUX PIPELINE")
    print("STEP 4 - OBJECT INSTANCING")
    print("=" * 60)

    # ----------------------------------------
    # Load semantic point cloud
    # ----------------------------------------

    data = load_semantic_point_cloud(
        "DATA/indoor_room_labelled.csv"
    )

    print()
    print(
        f"Loaded {len(data):,} points"
    )

    print()
    print("Semantic classes:")
    print(
        data["semantic_label"]
        .value_counts()
    )

    # ----------------------------------------
    # Run DBSCAN object extraction
    # ----------------------------------------

    objects = extract_semantic_objects(
        data,
        eps=0.5,
        min_samples=10
    )
    # ========================================
# STEP 6: COMPUTE OBJECT FEATURES
# ========================================

    features = compute_object_features(
        objects
        )
    with open(
    "output/object_features.json",
    "w"
) as f:

        json.dump(
        features,
        f,
        indent=2,
        default=lambda x: x.tolist()
        if hasattr(x, "tolist")
        else x
    )

    print()
    print("=" * 60)
    print("OBJECT FEATURES COMPUTED")
    print("=" * 60)

    print(
    "File: output/object_features.json"
)

    # ----------------------------------------
    # Results
    # ----------------------------------------

    print()
    print("=" * 60)
    print("OBJECT EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Total objects found: "
        f"{len(objects)}"
    )

    # ----------------------------------------
    # Print object information
    # ----------------------------------------

    print()
    print("Extracted objects:")
    print("-" * 60)

    for object_key, obj in objects.items():

        print(
            f"{object_key:25s} "
            f"| class={obj['semantic_label']:10s} "
            f"| points={obj['point_count']:6d}"
        )

        print(
            f"    centroid: "
            f"{obj['centroid']}"
        )

        print(
            f"    min: "
            f"{obj['bounds']['min']}"
        )

        print(
            f"    max: "
            f"{obj['bounds']['max']}"
        )

    # ========================================
    # SAVE OBJECT INFORMATION FOR STEP 5
    # ========================================

    os.makedirs("output", exist_ok=True)
    os.makedirs("output/instance_points", exist_ok=True)

    objects_json = {}

    for object_key, obj in objects.items():

        objects_json[object_key] = {
            "semantic_label": obj["semantic_label"],
            "point_count": int(obj["point_count"]),

            "centroid": [
                float(x)
                for x in obj["centroid"]
            ],

            "bounds": {
                "min": [
                    float(x)
                    for x in obj["bounds"]["min"]
                ],

                "max": [
                    float(x)
                    for x in obj["bounds"]["max"]
                ]
            }
        }

        # Save this instance's raw points so Step 10 can
        # reconstruct a real mesh instead of a bounding-box cube.
        np.save(
            f"output/instance_points/{object_key}.npy",
            obj["points"][["x", "y", "z"]].values.astype(np.float32)
        )

    with open(
        "output/objects.json",
        "w"
    ) as f:

        json.dump(
            objects_json,
            f,
            indent=2
        )

    print()
    print("=" * 60)
    print("OBJECT DATA SAVED")
    print("=" * 60)
    print(
        f"Saved {len(objects_json)} objects"
    )
    print(
        "File: output/objects.json"
    )