# ============================================
# Florent Poux Tutorial
# Step 5 - Building the Scene Graph
# ============================================

import json
import math
import networkx as nx
import matplotlib.pyplot as plt


# ============================================
# STEP 1: Load objects
# ============================================

def load_objects(file_path):

    with open(file_path, "r") as f:
        objects = json.load(f)

    print()
    print("=" * 60)
    print("STEP 5 - SCENE GRAPH")
    print("=" * 60)

    print()
    print(f"Loaded {len(objects)} objects")

    return objects


# ============================================
# STEP 2: Create graph nodes
# ============================================

def create_scene_graph(objects):

    G = nx.Graph()

    for object_id, obj in objects.items():

        G.add_node(
            object_id,

            semantic_label=obj["semantic_label"],
            point_count=obj["point_count"],
            centroid=obj["centroid"],
            bounds=obj["bounds"]
        )

    return G


# ============================================
# STEP 3: Calculate distance between objects
# ============================================

def calculate_distance(obj1, obj2):

    c1 = obj1["centroid"]
    c2 = obj2["centroid"]

    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dz = c1[2] - c2[2]

    distance = math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )

    return distance


# ============================================
# STEP 4: Add spatial relationships
# ============================================

def add_spatial_relationships(
    G,
    objects,
    near_distance=2.0,
    vertical_threshold=0.5
):

    object_ids = list(objects.keys())

    print()
    print("=" * 60)
    print("CALCULATING SPATIAL RELATIONSHIPS")
    print("=" * 60)

    relationship_count = 0

    for i in range(len(object_ids)):

        for j in range(i + 1, len(object_ids)):

            id1 = object_ids[i]
            id2 = object_ids[j]

            obj1 = objects[id1]
            obj2 = objects[id2]

            # --------------------------------
            # Distance
            # --------------------------------

            distance = calculate_distance(
                obj1,
                obj2
            )

            # --------------------------------
            # NEAR relationship
            # --------------------------------

            if distance <= near_distance:

                G.add_edge(
                    id1,
                    id2,
                    relation="near",
                    distance=distance
                )

                print(
                    f"{id1:20s} "
                    f"-- near --> "
                    f"{id2:20s} "
                    f"distance={distance:.2f}"
                )

                relationship_count += 1

            # --------------------------------
            # ABOVE / BELOW relationship
            # --------------------------------

            z1 = obj1["centroid"][2]
            z2 = obj2["centroid"][2]

            if abs(z1 - z2) > vertical_threshold:

                if z1 > z2:

                    G.add_edge(
                        id1,
                        id2,
                        relation="above",
                        distance=distance
                    )

                else:

                    G.add_edge(
                        id1,
                        id2,
                        relation="above",
                        distance=distance
                    )

                relationship_count += 1

    print()
    print(
        f"Total relationships found: "
        f"{relationship_count}"
    )


# ============================================
# STEP 5: Print graph information
# ============================================

def print_graph_information(G):

    print()
    print("=" * 60)
    print("SCENE GRAPH INFORMATION")
    print("=" * 60)

    print(
        f"Nodes: {G.number_of_nodes()}"
    )

    print(
        f"Edges: {G.number_of_edges()}"
    )

    print()
    print("Relationships:")
    print("-" * 60)

    for node1, node2, data in G.edges(data=True):

        print(
            f"{node1:20s} "
            f"-- {data['relation']:6s} --> "
            f"{node2:20s} "
            f"distance={data['distance']:.2f}"
        )


# ============================================
# STEP 6: Visualize graph
# ============================================

def visualize_graph(G):

    plt.figure(figsize=(16, 12))

    pos = nx.spring_layout(
        G,
        seed=42
    )

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1000,
        font_size=7,
        font_weight="bold"
    )

    plt.title(
        "Scene Graph - Spatial Relationships"
    )

    plt.show()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    # Load objects from Step 4
    objects = load_objects(
        "output/objects.json"
    )

    # Create nodes
    G = create_scene_graph(
        objects
    )

    # Add relationships
    add_spatial_relationships(
        G,
        objects,
        near_distance=2.0,
        vertical_threshold=0.5
    )

    # Print results
    print_graph_information(G)

    # Visualize
    visualize_graph(G)