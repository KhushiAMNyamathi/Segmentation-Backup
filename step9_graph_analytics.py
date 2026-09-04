# ============================================
# Florent Poux Tutorial
# Step 9 - NetworkX Analytics
# ============================================

import json
import networkx as nx


# ============================================
# STEP 1: Load Scene Graph
# ============================================

def load_scene_graph(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)

    G = nx.node_link_graph(
        data,
        directed=True,
        edges="edges"
    )

    return G


# ============================================
# STEP 2: Analyze Scene Graph
# ============================================

def analyze_scene_graph(G):

    analysis = {

        "node_count":
            G.number_of_nodes(),

        "edge_count":
            G.number_of_edges(),

        "semantic_distribution":
            {},

        "relationship_types":
            {},

        "connected_components":
            nx.number_weakly_connected_components(G),

        "avg_degree":
            (
                sum(
                    dict(G.degree()).values()
                )
                / G.number_of_nodes()
                if G.number_of_nodes() > 0
                else 0
            )
    }

    # ----------------------------------------
    # Semantic distribution
    # ----------------------------------------

    for node, data in G.nodes(data=True):

        label = data.get(
            "semantic_label",
            "unknown"
        )

        analysis[
            "semantic_distribution"
        ][label] = (
            analysis[
                "semantic_distribution"
            ].get(label, 0) + 1
        )

    # ----------------------------------------
    # Relationship types
    # ----------------------------------------

    for _, _, data in G.edges(data=True):

        rel = data.get(
            "relationship",
            "unknown"
        )

        analysis[
            "relationship_types"
        ][rel] = (
            analysis[
                "relationship_types"
            ].get(rel, 0) + 1
        )

    return analysis


# ============================================
# STEP 3: Print Analytics
# ============================================

def print_analysis(analysis):

    print()
    print("=" * 60)
    print("STEP 9 - NETWORKX SCENE GRAPH ANALYTICS")
    print("=" * 60)

    print()

    print(
        f"Node count: "
        f"{analysis['node_count']}"
    )

    print(
        f"Edge count: "
        f"{analysis['edge_count']}"
    )

    print(
        f"Connected components: "
        f"{analysis['connected_components']}"
    )

    print(
        f"Average degree: "
        f"{analysis['avg_degree']:.2f}"
    )

    # ----------------------------------------
    # Semantic distribution
    # ----------------------------------------

    print()
    print("SEMANTIC DISTRIBUTION")
    print("-" * 60)

    for label, count in sorted(
        analysis[
            "semantic_distribution"
        ].items()
    ):

        print(
            f"{label:20s}: {count}"
        )

    # ----------------------------------------
    # Relationship distribution
    # ----------------------------------------

    print()
    print("RELATIONSHIP TYPES")
    print("-" * 60)

    for relation, count in sorted(
        analysis[
            "relationship_types"
        ].items()
    ):

        print(
            f"{relation:20s}: {count}"
        )


# ============================================
# STEP 4: Save Analytics
# ============================================

def save_analysis(
    analysis,
    file_path
):

    with open(
        file_path,
        "w"
    ) as f:

        json.dump(
            analysis,
            f,
            indent=2
        )

    print()
    print(
        f"Analytics saved to: "
        f"{file_path}"
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    # ----------------------------------------
    # Load graph
    # ----------------------------------------

    G = load_scene_graph(
        "output/scene_graph.json"
    )

    print()
    print(
        f"Loaded scene graph: "
        f"{G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    # ----------------------------------------
    # Analyze
    # ----------------------------------------

    analysis = analyze_scene_graph(G)

    # ----------------------------------------
    # Print
    # ----------------------------------------

    print_analysis(
        analysis
    )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    save_analysis(
        analysis,
        "output/scene_graph_analysis.json"
    )