# ============================================
# STEP 2 — Load PointNet++ Semantic Point Cloud
# (Replaces old CSV-based loader)
# ============================================
#
# INPUT  (from teammate's PointNet++ project, already copied here):
#   input/room.ply       -> XYZ coordinates
#   input/labels.npy     -> semantic class id per point
#
# OUTPUT (used by Step 4 - object instancing):
#   output/semantic_points.pkl   -> pandas DataFrame with x,y,z,semantic_id,semantic_label
#   output/semantic_points.csv   -> same data, human-readable

import os
import numpy as np
import pandas as pd
from plyfile import PlyData

# ----------------------------------------
# Semantic class mapping (from teammate's S3DIS-pretrained PointNet++)
# ----------------------------------------

CLASS_NAMES = {
    -1: "_filtered",
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "table",
    8: "chair",
    9: "sofa",
    10: "bookcase",
    11: "board",
    12: "clutter",
}

INPUT_PLY = "input/room.ply"
INPUT_LABELS = "input/labels.npy"
OUTPUT_DIR = "output"
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "semantic_points.pkl")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "semantic_points.csv")


def load_semantic_point_cloud(ply_file=INPUT_PLY, labels_file=INPUT_LABELS):
    """
    Load XYZ coordinates from room.ply and semantic labels from
    PointNet++ labels.npy. Returns one DataFrame row per point.
    """

    print()
    print("=" * 60)
    print("STEP 2: LOADING POINTNET++ SEMANTIC POINT CLOUD")
    print("=" * 60)

    # ---- Load PLY geometry ----
    if not os.path.exists(ply_file):
        raise FileNotFoundError(f"Could not find {ply_file}")

    ply = PlyData.read(ply_file)
    vertex = ply["vertex"]

    xyz = np.column_stack([
        vertex["x"],
        vertex["y"],
        vertex["z"],
    ]).astype(np.float32)

    # ---- Load PointNet++ labels ----
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Could not find {labels_file}")

    labels = np.load(labels_file)

    print(f"PLY points: {len(xyz):,}")
    print(f"Labels:     {len(labels):,}")

    # ---- Verify correspondence (do not proceed silently if mismatched) ----
    if len(xyz) != len(labels):
        raise ValueError(
            f"PLY point count ({len(xyz)}) and labels count "
            f"({len(labels)}) do not match!"
        )

    print("MATCH: True")

    # ---- Map numeric labels -> class names ----
    semantic_names = [
        CLASS_NAMES.get(int(label), "unknown")
        for label in labels
    ]

    unknown_count = semantic_names.count("unknown")
    if unknown_count > 0:
        unknown_ids = sorted(set(
            int(l) for l, n in zip(labels, semantic_names) if n == "unknown"
        ))
        print(f"WARNING: {unknown_count:,} points had unrecognized label ids: {unknown_ids}")

    # ---- Build DataFrame ----
    df = pd.DataFrame({
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "semantic_id": labels.astype(int),
        "semantic_label": semantic_names,
    })

    return df


def save_semantic_point_cloud(df):
    """
    Persist the loaded+labeled point cloud so Step 4 can load it
    without recomputing anything.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_pickle(OUTPUT_PKL)
    df.to_csv(OUTPUT_CSV, index=False)
    print()
    print(f"Saved: {OUTPUT_PKL}")
    print(f"Saved: {OUTPUT_CSV}")


def print_summary(df):
    print()
    print("Dataset loaded successfully!")
    print(f"Total points: {len(df):,}")
    print()
    print("Columns:", df.columns.tolist())
    print()
    print("Semantic class counts:")
    print(df["semantic_label"].value_counts())


if __name__ == "__main__":

    print("=" * 60)
    print("MY-SEGMENTATION PIPELINE")
    print("STEP 2: POINTNET++ SEMANTIC POINT CLOUD")
    print("=" * 60)

    demo_data = load_semantic_point_cloud()
    print_summary(demo_data)
    save_semantic_point_cloud(demo_data)

    print()
    print("Step 2 complete. Proceed to Step 4 (object instancing) only")
    print("after checking output/semantic_points.csv looks correct.")