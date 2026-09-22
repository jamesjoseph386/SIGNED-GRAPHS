import itertools
import matplotlib.pyplot as plt
import networkx as nx


def compute_gamma_R(num_vertices, pos_edges, neg_edges):
    """Computes gamma_R(S) for a signed graph using exact brute-force search.

    Returns float('inf') if the signed graph is non-admissible.
    """
    pos_adj = {v: [] for v in range(1, num_vertices + 1)}
    neg_adj = {v: [] for v in range(1, num_vertices + 1)}

    for u, v in pos_edges:
        pos_adj[u].append(v)
        pos_adj[v].append(u)

    for u, v in neg_edges:
        neg_adj[u].append(v)
        neg_adj[v].append(u)

    min_weight = float("inf")

    # Iterate over all 3^n function assignments: f(v) in {0, 1, 2}
    for assignment in itertools.product([0, 1, 2], repeat=num_vertices):
        f = {v + 1: assignment[v] for v in range(num_vertices)}

        # Condition 1: If f(v) == 0, exists u in N+(v) with f(u) == 2
        cond1 = True
        for v in range(1, num_vertices + 1):
            if f[v] == 0:
                if not any(f[u] == 2 for u in pos_adj[v]):
                    cond1 = False
                    break
        if not cond1:
            continue

        # Condition 2: f(v) + sum_{u in N+(v)} f(u) - sum_{w in N-(v)} f(w) >= 1
        cond2 = True
        for v in range(1, num_vertices + 1):
            pos_sum = sum(f[u] for u in pos_adj[v])
            neg_sum = sum(f[w] for w in neg_adj[v])
            if f[v] + pos_sum - neg_sum < 1:
                cond2 = False
                break
        if not cond2:
            continue

        # Valid RDF found
        weight = sum(f.values())
        if weight < min_weight:
            min_weight = weight

    return min_weight


def check_signed_tree_criticality(n, pos_edges, neg_edges):
    """Checks if a signed graph S is gamma_R-edge critical."""
    base_gamma = compute_gamma_R(n, pos_edges, neg_edges)

    if base_gamma == float("inf"):
        return False, base_gamma, "Non-admissible graph", {}

    existing_edges = set(
        tuple(sorted(e)) for e in (pos_edges + neg_edges)
    )
    all_vertices = list(range(1, n + 1))

    non_adjacent_pairs = [
        (u, v)
        for u, v in itertools.combinations(all_vertices, 2)
        if tuple(sorted((u, v))) not in existing_edges
    ]

    if not non_adjacent_pairs:
        return True, base_gamma, None, {}

    is_critical = True
    edge_results = {}
    failed_pair = None

    for u, v in non_adjacent_pairs:
        aug_pos_edges = pos_edges + [(u, v)]
        aug_gamma = compute_gamma_R(n, aug_pos_edges, neg_edges)
        edge_results[(u, v)] = aug_gamma

        # Edge criticality requires strict reduction: gamma_R(S + e) < gamma_R(S)
        if aug_gamma >= base_gamma:
            is_critical = False
            if failed_pair is None:
                failed_pair = (u, v)

    return is_critical, base_gamma, failed_pair, edge_results


def build_signed_graph_nx(n, pos_edges, neg_edges):
    """Builds a NetworkX graph with edge sign attributes."""
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for u, v in pos_edges:
        G.add_edge(u, v, sign=1)
    for u, v in neg_edges:
        G.add_edge(u, v, sign=-1)
    return G


def is_isomorphic_signed(G1, G2):
    """Checks if two signed graphs are isomorphic considering edge signs."""
    em = lambda e1, e2: e1["sign"] == e2["sign"]
    return nx.is_isomorphic(G1, G2, edge_match=em)


def plot_signed_tree_group(title, graph_list, num_vertices):
    """Draws a grid collection of signed trees using Matplotlib."""
    if not graph_list:
        print(f"No graphs to plot for category: {title}")
        return

    num_graphs = len(graph_list)
    cols = min(4, num_graphs)
    rows = (num_graphs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))

    # Flatten axes array for easy iteration if multi-cell
    if num_graphs == 1:
        axes = [axes]
    elif rows > 1 or cols > 1:
        axes = axes.flatten()

    for idx, item in enumerate(graph_list):
        ax = axes[idx]
        G = nx.Graph()
        G.add_nodes_from(range(1, num_vertices + 1))

        pos_edges = item["pos_edges"]
        neg_edges = item["neg_edges"]
        base_gamma = item["base_gamma"]

        pos = nx.circular_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color="skyblue", node_size=500, edgecolors="black", ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, font_weight="bold", font_size=10, ax=ax
        )

        # Draw positive edges (Green, Solid)
        if pos_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=pos_edges,
                style="solid",
                edge_color="green",
                width=2,
                ax=ax,
            )

        # Draw negative edges (Red, Dashed)
        if neg_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=neg_edges,
                style="dashed",
                edge_color="red",
                width=2,
                ax=ax,
            )

        subplot_title = f"Tree {idx + 1}\n$\\gamma_R = {base_gamma}$"
        if item.get("failed_pair"):
            subplot_title += f"\nFailed Edge: {item['failed_pair']}"

        ax.set_title(subplot_title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Hide any remaining empty subplot axes
    for idx in range(num_graphs, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def analyze_and_draw_signed_trees(n):
    """Generates, evaluates, prints, and draws signed trees on n vertices."""
    if n < 3:
        print(f"For n = {n}, trees have no non-adjacent vertex pairs.")
        return

    # 1. Generate all non-isomorphic trees on n vertices
    base_trees = list(nx.nonisomorphic_trees(n))

    # Relabel nodes to 1..n
    relabeled_trees = []
    for T in base_trees:
        mapping = {node: i + 1 for i, node in enumerate(T.nodes())}
        relabeled_trees.append(nx.relabel_nodes(T, mapping))

    # 2. Filter distinct non-isomorphic signed trees with 1 negative edge
    unique_signed_trees = []

    for T in relabeled_trees:
        tree_edges = list(T.edges())
        for neg_e in tree_edges:
            pos_e = [e for e in tree_edges if e != neg_e]
            candidate_G = build_signed_graph_nx(n, pos_e, [neg_e])

            if not any(
                is_isomorphic_signed(candidate_G, existing)
                for existing in unique_signed_trees
            ):
                unique_signed_trees.append(candidate_G)

    # 3. Classify each unique signed tree
    critical_trees = []
    non_critical_trees = []

    for G in unique_signed_trees:
        pos_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d["sign"] == 1
        ]
        neg_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d["sign"] == -1
        ]

        is_crit, base_gamma, failed_pair, details = (
            check_signed_tree_criticality(n, pos_edges, neg_edges)
        )

        item = {
            "pos_edges": pos_edges,
            "neg_edges": neg_edges,
            "base_gamma": base_gamma,
            "failed_pair": failed_pair,
            "details": details,
        }

        if is_crit:
            critical_trees.append(item)
        else:
            non_critical_trees.append(item)

    # 4. Print Summary
    print("=" * 70)
    print(f"  SIGNED TREES ANALYSIS FOR n = {n} (1 NEGATIVE EDGE)")
    print("=" * 70)
    print(f"Total Non-Isomorphic Signed Trees: {len(unique_signed_trees)}")
    print(f"  - Critical Signed Trees        : {len(critical_trees)}")
    print(f"  - Non-Critical Signed Trees    : {len(non_critical_trees)}\n")

    # 5. Draw Graphs
    plot_signed_tree_group(
        f"$\\gamma_R$-Edge Critical Signed Trees (n = {n})",
        critical_trees,
        n,
    )
    plot_signed_tree_group(
        f"Non-$\\gamma_R$-Edge Critical Signed Trees (n = {n})",
        non_critical_trees,
        n,
    )


if __name__ == "__main__":
    try:
        user_n = int(
            input("Enter the number of vertices for the trees (n >= 3): ")
        )
        analyze_and_draw_signed_trees(user_n)
    except ValueError:
        print("Please enter a valid integer.")