# ============================================
# Florent Poux Tutorial
# Step 8 - Building the Scene Graph
# ============================================

import json
import networkx as nx


# ============================================
# STEP 1: Load data
# ============================================

def load_json(file_path):

    with open(file_path, "r") as f:
        return json.load(f)


# ============================================
# STEP 2: Build Scene Graph
# ============================================

def build_scene_graph(
    objects,
    relationships,
    features
):

    # Directed graph
    G = nx.DiGraph()

    # ----------------------------------------
    # Add nodes with rich attributes
    # ----------------------------------------

    for obj_name, obj_data in objects.items():

        obj_features = (
            features
            .get(obj_name, {})
            .copy()
        )

        # Avoid duplicate semantic_label
        obj_features.pop("semantic_label", None)
        obj_features.pop("centroid", None)

        G.add_node(
            obj_name,

            semantic_label=
                obj_data["semantic_label"],

            centroid=
                obj_data["centroid"],

            point_count=
                obj_data["point_count"],

            **obj_features
        )

    # ----------------------------------------
    # Add relationship edges
    # ----------------------------------------

    for obj1, obj2, rel_type in relationships:

        G.add_edge(
            obj1,
            obj2,
            relationship=rel_type
        )

    return G


# ============================================
# STEP 3: Print graph information
# ============================================

def print_graph_information(G):

    print()
    print("=" * 60)
    print("STEP 8 - SCENE GRAPH")
    print("=" * 60)

    print()
    print(
        f"Nodes: {G.number_of_nodes()}"
    )

    print(
        f"Edges: {G.number_of_edges()}"
    )

    print()
    print("NODES")
    print("-" * 60)

    for node, data in G.nodes(data=True):

        print(
            f"{node:20s} "
            f"| {data['semantic_label']:12s} "
            f"| points={data['point_count']}"
        )

    print()
    print("EDGES")
    print("-" * 60)

    for obj1, obj2, data in G.edges(data=True):

        print(
            f"{obj1:20s} "
            f"-- {data['relationship']:10s} --> "
            f"{obj2}"
        )


# ============================================
# STEP 4: Save graph
# ============================================

def save_scene_graph(
    G,
    file_path
):

    data = nx.node_link_data(
        G,
        edges="edges"
    )

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
        f"Scene graph saved to: "
        f"{file_path}"
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    # ----------------------------------------
    # Load objects
    # ----------------------------------------

    objects = load_json(
        "output/objects.json"
    )

    # ----------------------------------------
    # Load features
    # ----------------------------------------

    features = load_json(
        "output/object_features.json"
    )

    # ----------------------------------------
    # Load relationships
    # ----------------------------------------

    relationship_data = load_json(
        "output/spatial_relationships.json"
    )

    # Convert JSON dictionaries back into
    # article-style tuples
    relationships = [
        (
            item["object1"],
            item["object2"],
            item["relationship"]
        )
        for item in relationship_data
    ]

    # ----------------------------------------
    # Build graph
    # ----------------------------------------

    G = build_scene_graph(
        objects,
        relationships,
        features
    )

    # ----------------------------------------
    # Print
    # ----------------------------------------

    print_graph_information(G)

    # ----------------------------------------
    # Save
    # ----------------------------------------

    save_scene_graph(
        G,
        "output/scene_graph.json"
    )