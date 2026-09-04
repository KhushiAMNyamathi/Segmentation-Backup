# ============================================
# Florent Poux Tutorial - Step 2
# Semantic Point Cloud Loading + Visualization
# ============================================

# -----------------------------
# Base libraries
# -----------------------------
import numpy as np
import pandas as pd
import json

# -----------------------------
# Graph / algorithm libraries
# -----------------------------
import networkx as nx
from sklearn.cluster import DBSCAN

# -----------------------------
# Visualization
# -----------------------------
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -----------------------------
# Optional OpenUSD
# -----------------------------
try:
    from pxr import Usd, UsdGeom, Sdf, Gf, UsdShade
    USD_AVAILABLE = True
except ImportError:
    print("USD not available. This is OK for now.")
    USD_AVAILABLE = False


# ============================================
# STEP 2: Load semantic point cloud
# ============================================

def load_semantic_point_cloud(
    file_path,
    column_name="semantic_label"
):
    """
    Load semantic point cloud data from CSV.
    """

    # Read CSV using semicolon delimiter
    df = pd.read_csv(
        file_path,
        delimiter=";"
    )

    print(f"Loaded CSV: {file_path}")
    print(f"Original number of points: {len(df):,}")

    # Semantic classes from the tutorial
    class_names = [
        "ceiling",
        "floor",
        "wall",
        "chair",
        "furniture",
        "table"
    ]

    # Map numerical labels to class names
    label_map = {
        float(i): class_names[i]
        for i in range(len(class_names))
    }

    df[column_name] = df[column_name].map(label_map)

    # Check for unknown labels
    unknown = df[column_name].isna().sum()

    if unknown > 0:
        print(
            f"Warning: {unknown:,} points "
            f"have unknown semantic labels."
        )

    # Tutorial samples 150,000 points
    sample_size = min(150000, len(df))

    df = df.sample(
        n=sample_size,
        random_state=1
    )

    return df


# ============================================
# Open3D Visualization
# ============================================

def visualize_semantic_pointcloud(
    df,
    point_size=3.0
):
    """
    Visualize semantic point cloud with
    different colors for each semantic class.
    """

    # Extract XYZ coordinates
    points = df[
        ["x", "y", "z"]
    ].values

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    # Get unique semantic classes
    unique_labels = (
        df["semantic_label"]
        .dropna()
        .unique()
    )

    # Create color mapping
    color_map = {}

    for i, label in enumerate(unique_labels):

        hue = i / len(unique_labels)

        color_map[label] = (
            plt.cm.tab10(hue)[:3]
        )

    # Assign colors to points
    colors = np.array([
        color_map[label]
        for label in df["semantic_label"]
    ])

    pcd.colors = (
        o3d.utility.Vector3dVector(colors)
    )

    # Create Open3D window
    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name="Semantic Point Cloud",
        width=1200,
        height=800
    )

    # Add point cloud
    vis.add_geometry(pcd)

    # Set point size
    render_option = (
        vis.get_render_option()
    )

    render_option.point_size = point_size

    # Print information
    print()
    print("=" * 50)
    print("SEMANTIC POINT CLOUD")
    print("=" * 50)

    print(
        f"Visualizing {len(points):,} points "
        f"with {len(unique_labels)} "
        f"semantic classes:"
    )

    for label in unique_labels:

        count = (
            df["semantic_label"] == label
        ).sum()

        print(
            f"  {label:12s}: "
            f"{count:,} points"
        )

    print("=" * 50)
    print("Close the Open3D window to continue.")

    # Start visualization
    vis.run()

    # Close window
    vis.destroy_window()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("FLORENT POUX - 3D SCENE GRAPH PIPELINE")
    print("STEP 2: SEMANTIC POINT CLOUD")
    print("=" * 50)

    # ----------------------------------------
    # Load dataset
    # ----------------------------------------

    raw_data = load_semantic_point_cloud(
        "DATA/indoor_room_labelled.csv"
    )

    # ----------------------------------------
    # Create demo dataset
    # ----------------------------------------

    demo_size = min(
        100000,
        len(raw_data)
    )

    demo_data = raw_data.sample(
        n=demo_size,
        random_state=1
    )

    # ----------------------------------------
    # Print dataset information
    # ----------------------------------------

    print()
    print("Dataset loaded successfully!")

    print(
        f"Raw sampled points: "
        f"{len(raw_data):,}"
    )

    print(
        f"Demo points: "
        f"{len(demo_data):,}"
    )

    print()
    print("Columns:")

    print(
        demo_data.columns.tolist()
    )

    print()
    print("Semantic classes:")

    print(
        demo_data[
            "semantic_label"
        ].value_counts()
    )

    # ----------------------------------------
    # Open3D visualization
    # ----------------------------------------

    visualize_semantic_pointcloud(
        demo_data,
        point_size=3.0
    )