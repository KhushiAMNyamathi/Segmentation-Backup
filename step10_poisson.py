# ============================================
# Florent Poux Tutorial
# Step 10 - OpenUSD Integration and Export
# (mesh reconstruction: alpha shape -> poisson -> convex hull)
# ============================================

import json
import os
import numpy as np
import networkx as nx
import open3d as o3d

from pxr import Usd, UsdGeom, Sdf, Gf

INSTANCE_POINTS_DIR = "output/instance_points"
MIN_POINTS_FOR_MESH = 30


# ============================================
# STEP 1: Load scene graph
# ============================================

def load_scene_graph(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()

    for node in data["nodes"]:
        node_id = node["id"]
        attributes = {
            key: value
            for key, value in node.items()
            if key != "id"
        }
        G.add_node(node_id, **attributes)

    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            relationship=edge["relationship"]
        )

    return G


# ============================================
# STEP 2: Mesh reconstruction helpers
# ============================================

def try_alpha_shape(pcd, alpha=0.3):
    try:
        mesh = (
            o3d.geometry.TriangleMesh
            .create_from_point_cloud_alpha_shape(pcd, alpha)
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        if len(mesh.triangles) >= 4:
            return mesh
    except Exception:
        pass

    return None


def try_poisson(pcd, depth=8, density_trim_quantile=0.02):
    """
    Poisson needs oriented normals (already computed by the caller).
    Poisson tends to produce smoother, more solid-looking meshes than
    alpha shape, but can balloon outward in areas with sparse/no
    points, so we trim away the lowest-density fraction of the mesh
    (the parts Poisson had to guess at most) before accepting it.
    """
    try:
        mesh, densities = (
            o3d.geometry.TriangleMesh
            .create_from_point_cloud_poisson(pcd, depth=depth)
        )

        densities = np.asarray(densities)
        if len(densities) > 0:
            threshold = np.quantile(densities, density_trim_quantile)
            vertices_to_remove = densities < threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

        if len(mesh.triangles) >= 4:
            return mesh
    except Exception:
        pass

    return None


def try_convex_hull(pcd):
    try:
        mesh, _ = pcd.compute_convex_hull()
        if mesh is not None and len(mesh.triangles) > 0:
            return mesh
    except Exception:
        pass

    return None


def reconstruct_mesh_from_points(points, min_points=MIN_POINTS_FOR_MESH):
    """
    Try to reconstruct a solid mesh from an object's raw points.
    Tries alpha shape first (tightest fit to sparse/partial data),
    then Poisson (smoother, fills small gaps), then convex hull
    (always succeeds, but can't represent concave shapes).

    Returns (vertices, faces, method_name) or None if nothing usable
    could be built (caller falls back to a bounding-box cube).
    """

    if len(points) < min_points:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    if len(pcd.points) < min_points:
        return None

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    for method_name, mesh in [
        # ("alpha_shape", try_alpha_shape(pcd, alpha=0.3)),
        ("poisson", try_poisson(pcd)),
        ("convex_hull", try_convex_hull(pcd)),
    ]:
        if mesh is not None:
            return np.asarray(mesh.vertices), np.asarray(mesh.triangles), method_name

    return None


def create_mesh_prim(stage, path, vertices, faces):
    """Create a UsdGeom.Mesh from reconstructed vertices/faces."""

    mesh = UsdGeom.Mesh.Define(stage, path)

    mesh.CreatePointsAttr(
        [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    )
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(
        [int(i) for tri in faces for i in tri]
    )
    mesh.CreateSubdivisionSchemeAttr("none")

    return mesh.GetPrim()


# ============================================
# STEP 3: Create USD object (mesh or cube fallback)
# ============================================

def create_usd_object(stage, object_id, data):

    centroid = data.get("centroid", [0.0, 0.0, 0.0])

    bounds = data.get(
        "bounds",
        {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    )

    min_bound = bounds["min"]
    max_bound = bounds["max"]

    safe_name = str(object_id).replace(" ", "_")
    path = f"/Scene/Geometry/{safe_name}"

    mesh_result = None
    points_file = os.path.join(INSTANCE_POINTS_DIR, f"{object_id}.npy")

    if os.path.exists(points_file):
        raw_points = np.load(points_file)
        mesh_result = reconstruct_mesh_from_points(raw_points)

    if mesh_result is not None:
        vertices, faces, method_name = mesh_result
        prim = create_mesh_prim(stage, path, vertices, faces)
        print(
            f"  {object_id}: mesh via {method_name} "
            f"({len(vertices)} verts, {len(faces)} tris)"
        )

    else:
        size_x = max(max_bound[0] - min_bound[0], 0.01)
        size_y = max(max_bound[1] - min_bound[1], 0.01)
        size_z = max(max_bound[2] - min_bound[2], 0.01)

        cube = UsdGeom.Cube.Define(stage, path)

        cube.AddTranslateOp().Set(
            Gf.Vec3d(float(centroid[0]), float(centroid[1]), float(centroid[2]))
        )
        cube.AddScaleOp().Set(
            Gf.Vec3d(size_x / 2.0, size_y / 2.0, size_z / 2.0)
        )

        prim = cube.GetPrim()
        print(f"  {object_id}: cube fallback (no/insufficient points)")

    prim.CreateAttribute(
        "semantic_label", Sdf.ValueTypeNames.String
    ).Set(str(data.get("semantic_label", "unknown")))

    prim.CreateAttribute(
        "object_id", Sdf.ValueTypeNames.String
    ).Set(str(object_id))

    prim.CreateAttribute(
        "point_count", Sdf.ValueTypeNames.Int
    ).Set(int(data.get("point_count", 0)))

    prim.CreateAttribute(
        "centroid", Sdf.ValueTypeNames.Double3
    ).Set(Gf.Vec3d(float(centroid[0]), float(centroid[1]), float(centroid[2])))

    for attribute_name in [
        "volume", "surface_area", "compactness", "height", "point_density"
    ]:
        if attribute_name in data:
            prim.CreateAttribute(
                attribute_name, Sdf.ValueTypeNames.Float
            ).Set(float(data[attribute_name]))

    return prim


# ============================================
# STEP 4: Add spatial relationships
# ============================================

def add_relationships_to_stage(stage, scene_graph):

    relationship_data = []

    for source, target, data in scene_graph.edges(data=True):

        relationship = data.get("relationship", "unknown")

        relationship_data.append({
            "source": source,
            "target": target,
            "relationship": relationship
        })

    scene_prim = stage.GetPrimAtPath("/Scene")

    scene_prim.CreateAttribute(
        "spatial_relationships", Sdf.ValueTypeNames.String
    ).Set(json.dumps(relationship_data))

    scene_prim.CreateAttribute(
        "relationship_count", Sdf.ValueTypeNames.Int
    ).Set(len(relationship_data))

    return relationship_data


# ============================================
# STEP 5: Create USD stage
# ============================================

def create_usd_stage(scene_graph, output_path):

    print()
    print("=" * 60)
    print("CREATING USD STAGE")
    print("=" * 60)

    stage = Usd.Stage.CreateNew(output_path)

    root_prim = stage.DefinePrim("/Scene", "Xform")
    stage.SetDefaultPrim(root_prim)

    UsdGeom.Scope.Define(stage, "/Scene/Geometry")

    print()
    print("Adding objects...")
    print("-" * 60)

    for object_id, data in scene_graph.nodes(data=True):
        create_usd_object(stage, object_id, data)

    print()
    print("Adding spatial relationships...")
    print("-" * 60)

    relationships = add_relationships_to_stage(stage, scene_graph)

    print(f"Added {len(relationships)} relationships")

    stage.Save()

    print()
    print("USD stage saved:")
    print(output_path)

    return True


# ============================================
# STEP 6: Main
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 10 - OPENUSD INTEGRATION AND EXPORT")
    print("=" * 60)

    input_file = "output/scene_graph.json"
    output_file = "output/scene_graph.usda"

    if not os.path.exists(input_file):
        print()
        print(f"ERROR: Input file not found:")
        print(input_file)
        raise SystemExit(1)

    print()
    print("Loading scene graph...")

    G = load_scene_graph(input_file)

    print(
        f"Loaded {G.number_of_nodes()} nodes and "
        f"{G.number_of_edges()} edges"
    )

    success = create_usd_stage(G, output_file)

    print()
    print("=" * 60)

    if success:
        print("STEP 10 COMPLETE")
        print()
        print("Created:")
        print(output_file)
    else:
        print("STEP 10 FAILED")

    print("=" * 60)