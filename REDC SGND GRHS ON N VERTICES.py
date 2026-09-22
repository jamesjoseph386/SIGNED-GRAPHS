"""
all_single_negative_edge_critical.py
=====================================

For a user-supplied number of vertices n, this program considers EVERY
signed graph on n vertices that has exactly one negative edge (all other
edges positive), and classifies each one as gamma_R-edge-critical or not.

Output: two lists -- the signed graphs that ARE gamma_R-edge-critical, and
the signed graphs that are NOT.

--------------------------------------------------------------------------
IMPORTANT: what "every signed graph" means here, and why
--------------------------------------------------------------------------
The number of LABELLED simple graphs on n vertices is 2^C(n,2), and for
each one there are up to C(n,2) choices of which edge is the negative
edge.  This blows up far too fast to enumerate directly even for modest n
(2^28 for n = 8), and the overwhelming majority of those labelled graphs
are just relabellings of one another -- they give the exact same answer
(critical or not) since gamma_R-edge-criticality is an isomorphism
invariant of the signed graph.

So this program instead enumerates all signed graphs with one negative
edge on n vertices UP TO ISOMORPHISM:

  1. Generate every non-isomorphic (optionally: connected) simple graph G
     on n vertices, using a Weisfeiler-Lehman hash to bucket candidates
     and networkx's exact is_isomorphic() to confirm/deduplicate within
     each bucket.

  2. For each such G, compute its automorphism group, and partition its
     edges into ORBITS under that group.  Placing the negative edge on
     two edges in the same orbit gives ISOMORPHIC signed graphs, so only
     one representative edge per orbit needs to be checked.

  3. For every (graph, negative-edge-orbit-representative) pair, build
     the signed graph and run the same exact gamma_R / edge-criticality
     machinery used in the earlier scripts in this series.

This is still exponential (both in generating graphs and in the 3^n RDF
brute force applied to each one), but it is the smallest search that is
actually complete and non-redundant, and it is very comfortable up to
about n = 6, workable with patience at n = 7, and NOT recommended beyond
that with this exact/brute-force approach.

Usage
-----
Run interactively:

    python3 all_single_negative_edge_critical.py

or import and call directly:

    from all_single_negative_edge_critical import scan_all_signed_graphs

    result = scan_all_signed_graphs(n=6, connected_only=True)
"""

from itertools import combinations, product
import warnings

import networkx as nx

warnings.filterwarnings("ignore", category=UserWarning, module="networkx")


# =========================================================================
# Part 1: the exact RDF / gamma_R / edge-criticality machinery
#         (self-contained, same semantics as the earlier scripts)
# =========================================================================

def build_signed_adjacency(vertices, pos_edges, neg_edges):
    vertex_set = set(vertices)
    adj = {v: [] for v in vertices}

    def add_edge(u, v, sign):
        adj[u].append((v, sign))
        adj[v].append((u, sign))

    for (u, v) in pos_edges:
        add_edge(u, v, +1)
    for (u, v) in neg_edges:
        add_edge(u, v, -1)

    return adj


def is_valid_rdf(f, adj):
    for v, neighbours in adj.items():
        fv = f[v]
        balance = fv + sum(sign * f[u] for (u, sign) in neighbours)
        if balance < 1:
            return False
        if fv == 0:
            if not any(sign == 1 and f[u] == 2 for (u, sign) in neighbours):
                return False
    return True


def all_rdfs(vertices, adj):
    n = len(vertices)
    for values in product((0, 1, 2), repeat=n):
        f = dict(zip(vertices, values))
        if is_valid_rdf(f, adj):
            yield f


def gamma_R_weight(vertices, adj):
    """Roman domination number only (no RDF list kept) -- used for the
    many S+uv checks where only the weight is needed. Returns None if no
    RDF exists (e.g. an isolated negative edge)."""
    best = None
    for values in product((0, 1, 2), repeat=len(vertices)):
        f = dict(zip(vertices, values))
        if is_valid_rdf(f, adj):
            w = sum(values)
            if best is None or w < best:
                best = w
    return best


def find_all_minimum_rdfs(vertices, pos_edges, neg_edges):
    """Returns (gamma_R, list_of_minimum_rdfs) where each RDF is a list
    of (vertex, value) tuples.  gamma_R is None if infeasible."""
    adj = build_signed_adjacency(vertices, pos_edges, neg_edges)

    best_weight = None
    best_rdfs = []
    for f in all_rdfs(vertices, adj):
        w = sum(f.values())
        if best_weight is None or w < best_weight:
            best_weight = w
            best_rdfs = [f]
        elif w == best_weight:
            best_rdfs.append(f)

    if best_weight is None:
        return None, []

    rdfs_as_tuples = [[(v, f[v]) for v in vertices] for f in best_rdfs]
    return best_weight, rdfs_as_tuples


def non_adjacent_pairs(vertices, pos_edges, neg_edges):
    existing = {frozenset(e) for e in pos_edges} | {frozenset(e) for e in neg_edges}
    return [(u, v) for u, v in combinations(vertices, 2)
            if frozenset((u, v)) not in existing]


def analyze_edge_criticality(vertices, pos_edges, neg_edges):
    """
    Returns a dict:
        'gamma_R'          : gamma_R(S), or None if S itself is infeasible
        'minimum_rdfs'     : list of minimum RDFs (only if feasible)
        'is_edge_critical' : bool, or None if S is infeasible
        'failing_pairs'    : list of pairs failing to strictly decrease
                              gamma_R (empty iff edge-critical)
    """
    gamma_R, minimum_rdfs = find_all_minimum_rdfs(vertices, pos_edges, neg_edges)

    if gamma_R is None:
        return {
            'gamma_R': None,
            'minimum_rdfs': [],
            'is_edge_critical': None,
            'failing_pairs': None,
        }

    failing_pairs = []
    for (u, v) in non_adjacent_pairs(vertices, pos_edges, neg_edges):
        adj_new = build_signed_adjacency(vertices, pos_edges + [(u, v)], neg_edges)
        gamma_new = gamma_R_weight(vertices, adj_new)
        # gamma_new is never None here: adding a positive edge can only
        # preserve feasibility of S's own minimum RDF (see note below).
        if not (gamma_new < gamma_R):
            failing_pairs.append((u, v))

    return {
        'gamma_R': gamma_R,
        'minimum_rdfs': minimum_rdfs,
        'is_edge_critical': len(failing_pairs) == 0,
        'failing_pairs': failing_pairs,
    }


# =========================================================================
# Part 2: generating all non-isomorphic graphs on n vertices, with the
#         automorphism group and edge orbits of each one
# =========================================================================

def _is_connected(vertices, edges):
    if not vertices:
        return True
    adj = {v: set() for v in vertices}
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen = {vertices[0]}
    stack = [vertices[0]]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == len(vertices)


def generate_nonisomorphic_graphs(vertices, connected_only=True):
    """
    Yield one networkx.Graph representative per isomorphism class of
    simple graphs on `vertices` (optionally restricted to connected
    graphs).  Uses a Weisfeiler-Lehman hash to bucket candidates cheaply,
    then nx.is_isomorphic for an exact check within each bucket.
    """
    possible_edges = list(combinations(vertices, 2))
    m = len(possible_edges)

    buckets = {}  # wl_hash -> list of representative graphs

    for bits in range(2 ** m):
        edges = [possible_edges[i] for i in range(m) if (bits >> i) & 1]

        if connected_only and not _is_connected(vertices, edges):
            continue

        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)

        h = nx.weisfeiler_lehman_graph_hash(G)
        bucket = buckets.setdefault(h, [])

        if any(nx.is_isomorphic(G, rep) for rep in bucket):
            continue

        bucket.append(G)
        yield G


def automorphism_group(G):
    """Return the list of automorphisms of G, each as a dict
    {vertex: vertex}, found via VF2 (G matched against itself)."""
    GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
    return list(GM.isomorphisms_iter())


def edge_orbits(G, automorphisms):
    """
    Partition E(G) into orbits under the automorphism group.  Returns a
    list of representative edges (u, v), one per orbit -- placing the
    negative edge on any edge in the same orbit gives an isomorphic
    signed graph, so only these representatives need to be checked.
    """
    edges = [tuple(e) for e in G.edges()]
    edge_set = {frozenset(e) for e in edges}
    assigned = set()
    reps = []

    for e in edges:
        fe = frozenset(e)
        if fe in assigned:
            continue
        reps.append(e)
        for phi in automorphisms:
            mapped = frozenset((phi[e[0]], phi[e[1]]))
            if mapped in edge_set:
                assigned.add(mapped)

    return reps


# =========================================================================
# Part 3: putting it together
# =========================================================================

def scan_all_signed_graphs(n, connected_only=True, vertex_labels=None):
    """
    Enumerate every signed graph with exactly one negative edge on n
    vertices, up to isomorphism, and classify each as gamma_R-edge-
    critical or not.

    Parameters
    ----------
    n              : number of vertices
    connected_only : if True (default), only connected underlying graphs
                      are considered
    vertex_labels  : optional list of n labels to use for display
                      (defaults to 1..n)

    Returns
    -------
    dict with keys:
        'critical'   : list of result-records (see below) that ARE
                        gamma_R-edge-critical
        'non_critical': list of result-records that are NOT
        'infeasible' : list of result-records for signed graphs that
                        have no RDF at all (gamma_R undefined) -- only
                        possible when connected_only=False, or for very
                        small n
        'n_graph_classes' : number of non-isomorphic underlying graphs
                             considered

    Each result-record is a dict:
        'vertices'    : list of vertex labels
        'pos_edges'   : list of positive edges
        'neg_edge'    : the single negative edge, as a tuple
        'gamma_R'     : gamma_R(S), or None if infeasible
        'minimum_rdfs': list of minimum RDFs (list of (vertex,value)
                         tuples), or [] if infeasible
        'is_edge_critical' : True / False / None (None = infeasible)
        'failing_pairs'     : list of pairs that fail to strictly
                               decrease gamma_R (None if infeasible)
    """
    vertices = list(vertex_labels) if vertex_labels else list(range(1, n + 1))

    critical = []
    non_critical = []
    infeasible = []
    n_graph_classes = 0

    for G in generate_nonisomorphic_graphs(vertices, connected_only=connected_only):
        n_graph_classes += 1
        all_edges = [tuple(e) for e in G.edges()]

        autos = automorphism_group(G)
        reps = edge_orbits(G, autos)

        for neg_edge in reps:
            pos_edges = [e for e in all_edges if frozenset(e) != frozenset(neg_edge)]
            neg_edges = [neg_edge]

            analysis = analyze_edge_criticality(vertices, pos_edges, neg_edges)

            record = {
                'vertices': vertices,
                'pos_edges': pos_edges,
                'neg_edge': neg_edge,
                'gamma_R': analysis['gamma_R'],
                'minimum_rdfs': analysis['minimum_rdfs'],
                'is_edge_critical': analysis['is_edge_critical'],
                'failing_pairs': analysis['failing_pairs'],
            }

            if analysis['is_edge_critical'] is None:
                infeasible.append(record)
            elif analysis['is_edge_critical']:
                critical.append(record)
            else:
                non_critical.append(record)

    return {
        'critical': critical,
        'non_critical': non_critical,
        'infeasible': infeasible,
        'n_graph_classes': n_graph_classes,
    }


# =========================================================================
# Pretty-printing
# =========================================================================

def _format_record(record, index):
    lines = []
    lines.append(f"  [{index}] positive edges: {record['pos_edges']}")
    lines.append(f"       negative edge : {record['neg_edge']}")
    if record['gamma_R'] is None:
        lines.append("       gamma_R       : INFEASIBLE (no RDF exists for this signed graph)")
    else:
        lines.append(f"       gamma_R(S)    : {record['gamma_R']}")
        lines.append(f"       # minimum RDFs: {len(record['minimum_rdfs'])}")
        if record['is_edge_critical'] is False:
            lines.append(f"       failing pairs : {record['failing_pairs']}")
    return "\n".join(lines)


def summarize(n, connected_only=True, vertex_labels=None, show_rdfs=False):
    result = scan_all_signed_graphs(n, connected_only=connected_only, vertex_labels=vertex_labels)

    critical = result['critical']
    non_critical = result['non_critical']
    infeasible = result['infeasible']

    total = len(critical) + len(non_critical) + len(infeasible)

    print()
    print("=" * 78)
    print(f" Signed graphs with a single negative edge on {n} vertices, up to isomorphism")
    print(f" ({'connected graphs only' if connected_only else 'connected AND disconnected graphs'})")
    print("=" * 78)
    print(f"Non-isomorphic underlying graphs considered : {result['n_graph_classes']}")
    print(f"Distinct signed graphs checked (up to iso.)  : {total}")
    print(f"  -> gamma_R-edge-critical                   : {len(critical)}")
    print(f"  -> NOT gamma_R-edge-critical                : {len(non_critical)}")
    if infeasible:
        print(f"  -> infeasible (no RDF exists)               : {len(infeasible)}")
    print()

    print("-" * 78)
    print(f"GAMMA_R-EDGE-CRITICAL signed graphs ({len(critical)}):")
    print("-" * 78)
    if not critical:
        print("  (none)")
    for i, r in enumerate(critical, 1):
        print(_format_record(r, i))
        if show_rdfs:
            for j, rdf in enumerate(r['minimum_rdfs'], 1):
                print(f"         min RDF #{j}: {rdf}")
        print()

    print("-" * 78)
    print(f"NOT gamma_R-edge-critical signed graphs ({len(non_critical)}):")
    print("-" * 78)
    if not non_critical:
        print("  (none)")
    for i, r in enumerate(non_critical, 1):
        print(_format_record(r, i))
        if show_rdfs:
            for j, rdf in enumerate(r['minimum_rdfs'], 1):
                print(f"         min RDF #{j}: {rdf}")
        print()

    if infeasible:
        print("-" * 78)
        print(f"INFEASIBLE signed graphs ({len(infeasible)}) -- no RDF exists at all:")
        print("-" * 78)
        for i, r in enumerate(infeasible, 1):
            print(_format_record(r, i))
        print()

    print("=" * 78)
    print()

    return result


# =========================================================================
# Interactive command-line interface
# =========================================================================

def _estimate_cost(n):
    """Rough estimate of labelled-graph-generation cost, for the warning
    message (the dominant up-front cost before isomorphism reduction)."""
    from math import comb
    return 2 ** comb(n, 2)


def run_interactive():
    print("=" * 78)
    print(" Scan ALL signed graphs with a single negative edge on n vertices")
    print(" for gamma_R-edge-criticality (up to isomorphism)")
    print("=" * 78)
    print()

    n = int(input("Number of vertices n: ").strip())

    conn_raw = input(
        "Restrict to CONNECTED underlying graphs only? [Y/n]: "
    ).strip().lower()
    connected_only = (conn_raw != "n")

    labels_raw = input(
        f"Vertex labels (comma/space separated, or Enter for 1..{n}): "
    ).strip()
    if labels_raw:
        vertex_labels = [tok.strip() for tok in labels_raw.replace(",", " ").split() if tok.strip()]
        if len(vertex_labels) != n:
            raise ValueError(f"You said {n} vertices but gave {len(vertex_labels)} labels.")
    else:
        vertex_labels = None

    est = _estimate_cost(n)
    if n >= 7:
        print(
            f"\nWarning: generating all labelled graphs on {n} vertices before "
            f"isomorphism-reduction involves checking 2^C({n},2) = {est:,} candidate "
            f"edge sets. This WILL be slow (likely many minutes or more). "
            f"n <= 6 is comfortable; n = 7 needs patience; n >= 8 is not "
            f"recommended with this exact/brute-force approach.\n"
        )
    elif n >= 5:
        print(f"\n(This will check 2^C({n},2) = {est:,} candidate edge sets before "
              f"isomorphism-reduction -- should take from well under a second "
              f"to a few seconds.)\n")

    show_raw = input("Show all minimum RDFs for each signed graph too? [y/N]: ").strip().lower()
    show_rdfs = (show_raw == "y")

    summarize(n, connected_only=connected_only, vertex_labels=vertex_labels, show_rdfs=show_rdfs)


# =========================================================================

if __name__ == "__main__":
    run_interactive()