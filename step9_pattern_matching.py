# ============================================
# Florent Poux Tutorial
# Step 9 - Advanced Scene Graph Pattern Matching
# ============================================

import json
import numpy as np
import networkx as nx


# ============================================
# STEP 1: Load scene graph
# ============================================

def load_scene_graph(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Add nodes
    for node in data["nodes"]:

        node_id = node["id"]

        attributes = {
            key: value
            for key, value in node.items()
            if key != "id"
        }

        G.add_node(node_id, **attributes)

    # Add edges
    for edge in data["edges"]:

        G.add_edge(
            edge["source"],
            edge["target"],
            relationship=edge["relationship"]
        )

    return G


# ============================================
# STEP 2: Create flexible pattern matcher
# ============================================

def create_pattern_matcher(
    scene_graph,
    pattern_graph,
    node_attr="semantic_label",
    edge_attr="relationship"
):

    def semantic_node_match(node1_attrs, node2_attrs):

        label1 = node1_attrs.get(
            node_attr,
            "unknown"
        )

        label2 = node2_attrs.get(
            node_attr,
            "unknown"
        )

        # Exact match
        if label1 == label2:
            return True

        # Chair / seat / stool equivalence
        furniture_group = {
            "chair",
            "seat",
            "stool"
        }

        # Table / desk / surface equivalence
        surface_group = {
            "table",
            "desk",
            "surface"
        }

        if (
            label1 in furniture_group
            and label2 in furniture_group
        ):
            return True

        if (
            label1 in surface_group
            and label2 in surface_group
        ):
            return True

        return False

    def relationship_edge_match(
        edge1_attrs,
        edge2_attrs
    ):

        rel1 = edge1_attrs.get(
            edge_attr,
            "unknown"
        )

        rel2 = edge2_attrs.get(
            edge_attr,
            "unknown"
        )

        # Exact match
        if rel1 == rel2:
            return True

        # Proximity relationships
        proximity_group = {
            "adjacent",
            "near",
            "close"
        }

        # Support / vertical relationships
        support_group = {
            "above",
            "on",
            "supported_by"
        }

        if (
            rel1 in proximity_group
            and rel2 in proximity_group
        ):
            return True

        if (
            rel1 in support_group
            and rel2 in support_group
        ):
            return True

        return False

    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        scene_graph,
        pattern_graph,
        node_match=semantic_node_match,
        edge_match=relationship_edge_match
    )

    return matcher


# ============================================
# STEP 3: Meeting room pattern
# ============================================

def create_meeting_room_pattern():

    pattern = nx.DiGraph()

    # Central table
    pattern.add_node(
        "table_1",
        semantic_label="table"
    )

    # Three chairs
    pattern.add_node(
        "chair_1",
        semantic_label="chair"
    )

    pattern.add_node(
        "chair_2",
        semantic_label="chair"
    )

    pattern.add_node(
        "chair_3",
        semantic_label="chair"
    )

    # Chairs adjacent to table
    pattern.add_edge(
        "chair_1",
        "table_1",
        relationship="adjacent"
    )

    pattern.add_edge(
        "chair_2",
        "table_1",
        relationship="adjacent"
    )

    pattern.add_edge(
        "chair_3",
        "table_1",
        relationship="adjacent"
    )

    return pattern


# ============================================
# STEP 4: Calculate confidence
# ============================================

def calculate_pattern_confidence(
    scene_graph,
    node_mapping
):

    confidence_factors = []

    positions = []

    for scene_node in node_mapping.values():

        centroid = scene_graph.nodes[
            scene_node
        ].get(
            "centroid",
            [0, 0, 0]
        )

        positions.append(
            centroid
        )

    # ----------------------------------------
    # Geometric consistency
    # ----------------------------------------

    if len(positions) > 1:

        distances = []

        for i, pos1 in enumerate(positions):

            for pos2 in positions[i + 1:]:

                distance = np.linalg.norm(
                    np.array(pos1)
                    - np.array(pos2)
                )

                distances.append(
                    distance
                )

        avg_distance = np.mean(
            distances
        )

        spatial_confidence = max(
            0,
            1.0 - avg_distance / 5.0
        )

        confidence_factors.append(
            spatial_confidence
        )

    # ----------------------------------------
    # Semantic consistency
    # ----------------------------------------

    semantic_confidence = 1.0

    confidence_factors.append(
        semantic_confidence
    )

    return float(
        np.mean(confidence_factors)
    )


# ============================================
# STEP 5: Find patterns
# ============================================

def find_spatial_patterns(
    scene_graph,
    pattern_graph
):

    matcher = create_pattern_matcher(
        scene_graph,
        pattern_graph
    )

    pattern_matches = []

    for match in matcher.subgraph_isomorphisms_iter():

        scene_objects = list(
            match.values()
        )

        # ------------------------------------
        # Calculate pattern center
        # ------------------------------------

        centroids = []

        for obj_id in scene_objects:

            centroid = scene_graph.nodes[
                obj_id
            ].get(
                "centroid",
                [0, 0, 0]
            )

            centroids.append(
                centroid
            )

        if centroids:

            pattern_center = np.mean(
                centroids,
                axis=0
            )

            confidence = (
                calculate_pattern_confidence(
                    scene_graph,
                    match
                )
            )

            pattern_matches.append({

                "objects": scene_objects,

                "mapping": match,

                "center": (
                    pattern_center.tolist()
                ),

                "confidence": confidence
            })

    return pattern_matches


# ============================================
# STEP 6: Main
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 9 - ADVANCED SCENE GRAPH PATTERN MATCHING")
    print("=" * 60)

    # ----------------------------------------
    # Load scene graph
    # ----------------------------------------

    scene_graph = load_scene_graph(
        "output/scene_graph.json"
    )

    print()
    print(
        f"Loaded scene graph: "
        f"{scene_graph.number_of_nodes()} nodes, "
        f"{scene_graph.number_of_edges()} edges"
    )

    # ----------------------------------------
    # Create meeting room pattern
    # ----------------------------------------

    pattern = create_meeting_room_pattern()

    print()
    print(
        f"Meeting room pattern: "
        f"{pattern.number_of_nodes()} nodes, "
        f"{pattern.number_of_edges()} edges"
    )

    # ----------------------------------------
    # Search scene
    # ----------------------------------------

    matches = find_spatial_patterns(
        scene_graph,
        pattern
    )

    # ----------------------------------------
    # Print results
    # ----------------------------------------

    print()
    print("=" * 60)
    print("PATTERN MATCHING RESULTS")
    print("=" * 60)

    print()
    print(
        f"Meeting room patterns found: "
        f"{len(matches)}"
    )

    for i, match in enumerate(matches):

        print()
        print(
            f"Match {i + 1}"
        )

        print(
            f"  Objects: "
            f"{match['objects']}"
        )

        print(
            f"  Center: "
            f"{match['center']}"
        )

        print(
            f"  Confidence: "
            f"{match['confidence']:.3f}"
        )

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    with open(
        "output/pattern_matches.json",
        "w"
    ) as f:

        json.dump(
            matches,
            f,
            indent=2
        )

    print()
    print(
        "Pattern matches saved to: "
        "output/pattern_matches.json"
    )