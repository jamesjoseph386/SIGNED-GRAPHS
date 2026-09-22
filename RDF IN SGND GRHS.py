"""
signed_rdf.py
=============

Enumerate ALL minimum Roman Dominating Functions (RDFs) of a signed graph.

Definitions used (matching the standard signed-graph Roman domination setup):

    A signed graph S = (G, sigma) has each edge uv labelled sigma(uv) in {+1,-1}.

    A function f : V -> {0, 1, 2} is a Roman Dominating Function (RDF) of S if:

      (C1) every vertex v with f(v) = 0 has a neighbour u with f(u) = 2
           and sigma(uv) = +1  (a "positive 2-neighbour"), and

      (C2) for every vertex v,
                f(v) + sum_{u in N(v)} sigma(uv) * f(u)  >= 1.

    The weight of f is sum_v f(v).  The Roman domination number gamma_R(S)
    is the minimum weight over all RDFs, and a *minimum* RDF is one that
    attains this minimum.

This script brute-forces all 3^n functions f: V -> {0,1,2} (n = |V|), keeps
the ones that are valid RDFs, and returns every one of minimum weight.
Brute force is exact and simple, but exponential -- it is fine for graphs
up to about 16-18 vertices on an ordinary machine; beyond that it will be
slow (3^18 ~ 387 million).

Usage
-----
Run interactively:

    python3 signed_rdf.py

and answer the prompts, OR import and call the functions directly:

    from signed_rdf import find_all_minimum_rdfs

    vertices = ['a', 'b', 'c', 'd']
    pos_edges = [('a', 'b'), ('b', 'c')]
    neg_edges = [('c', 'd')]

    weight, rdfs = find_all_minimum_rdfs(vertices, pos_edges, neg_edges)
"""

from itertools import product
from collections import defaultdict


# --------------------------------------------------------------------------- #
# Core graph representation
# --------------------------------------------------------------------------- #

def build_signed_adjacency(vertices, pos_edges, neg_edges):
    """
    Build an adjacency structure: adj[v] = list of (neighbour, sign) pairs,
    sign is +1 or -1.  Also validates the input (no duplicate / conflicting
    edges, no self loops, all endpoints are declared vertices).
    """
    vertex_set = set(vertices)
    adj = {v: [] for v in vertices}
    seen_edges = {}

    def add_edge(u, v, sign):
        if u not in vertex_set or v not in vertex_set:
            raise ValueError(f"Edge ({u}, {v}) refers to an unknown vertex.")
        if u == v:
            raise ValueError(f"Self-loops are not allowed: ({u}, {v}).")
        key = frozenset((u, v))
        if key in seen_edges:
            raise ValueError(
                f"Edge {u}-{v} is specified more than once "
                f"(as sign {seen_edges[key]} and sign {sign})."
            )
        seen_edges[key] = sign
        adj[u].append((v, sign))
        adj[v].append((u, sign))

    for (u, v) in pos_edges:
        add_edge(u, v, +1)
    for (u, v) in neg_edges:
        add_edge(u, v, -1)

    return adj


# --------------------------------------------------------------------------- #
# RDF validity check
# --------------------------------------------------------------------------- #

def is_valid_rdf(f, adj):
    """
    f : dict {vertex: value in {0,1,2}}
    adj : dict {vertex: [(neighbour, sign), ...]}

    Returns True iff f satisfies (C1) and (C2).
    """
    for v, neighbours in adj.items():
        fv = f[v]

        # (C2) local balance condition -- holds for every vertex.
        balance = fv + sum(sign * f[u] for (u, sign) in neighbours)
        if balance < 1:
            return False

        # (C1) only constrains vertices currently valued 0.
        if fv == 0:
            has_positive_two_neighbour = any(
                sign == 1 and f[u] == 2 for (u, sign) in neighbours
            )
            if not has_positive_two_neighbour:
                return False

    return True


# --------------------------------------------------------------------------- #
# Brute-force search over all f : V -> {0,1,2}
# --------------------------------------------------------------------------- #

def all_rdfs(vertices, adj):
    """
    Generator yielding every valid RDF (as a dict) of the signed graph,
    found by exhaustive search over all 3^n assignments.
    """
    n = len(vertices)
    for values in product((0, 1, 2), repeat=n):
        f = dict(zip(vertices, values))
        if is_valid_rdf(f, adj):
            yield f


def _dict_to_tuple_list(f, vertices):
    """Convert an internal {vertex: value} dict into an ordered list of
    (vertex, value) tuples, following the order given in `vertices`."""
    return [(v, f[v]) for v in vertices]


def find_all_minimum_rdfs(vertices, pos_edges, neg_edges):
    """
    Main entry point.

    Parameters
    ----------
    vertices  : list of vertex labels (hashable, e.g. strings or ints)
    pos_edges : list of (u, v) pairs -- positive edges
    neg_edges : list of (u, v) pairs -- negative edges

    Returns
    -------
    (gamma_R, minimum_rdfs)
        gamma_R        : the Roman domination number (int)
        minimum_rdfs   : list of RDFs, where each RDF is itself a list of
                          (vertex, value) tuples -- one tuple per vertex,
                          in the same order as `vertices`.

                          Example (for vertices = ['a', 'b', 'c']):
                              [ [('a', 0), ('b', 2), ('c', 0)],
                                [('a', 2), ('b', 0), ('c', 1)], ... ]
    """
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
        raise RuntimeError(
            "No valid Roman dominating function exists for this signed "
            "graph (this should not happen for a graph with no isolated "
            "vertices lacking a way to satisfy (C2))."
        )

    # Convert every minimum RDF from an internal dict into the requested
    # "list of (vertex, value) tuples" format before returning.
    best_rdfs_as_tuples = [_dict_to_tuple_list(f, vertices) for f in best_rdfs]

    return best_weight, best_rdfs_as_tuples


# --------------------------------------------------------------------------- #
# Pretty-printing helpers
# --------------------------------------------------------------------------- #

def format_rdf(rdf_as_tuples):
    """rdf_as_tuples: a list of (vertex, value) tuples, e.g.
    [('a', 0), ('b', 2), ('c', 0)]."""
    return str(rdf_as_tuples)


def summarize(vertices, pos_edges, neg_edges):
    gamma_R, rdfs = find_all_minimum_rdfs(vertices, pos_edges, neg_edges)

    print()
    print(f"Number of vertices : {len(vertices)}")
    print(f"Positive edges     : {pos_edges}")
    print(f"Negative edges     : {neg_edges}")
    print("-" * 60)
    print(f"gamma_R(S) (Roman domination number) = {gamma_R}")
    print(f"Number of minimum RDFs                = {len(rdfs)}")
    print("-" * 60)
    for i, rdf in enumerate(rdfs, 1):
        print(f"  Minimum RDF #{i}: {format_rdf(rdf)}")
    print()

    # Also show, per vertex, the set of values it takes across all minimum
    # RDFs -- useful for spotting "pinned" vertices (forced to one value
    # in every minimum RDF) versus "flexible" ones.
    value_range = defaultdict(set)
    for rdf in rdfs:
        for (v, val) in rdf:
            value_range[v].add(val)

    print("Value(s) taken by each vertex across all minimum RDFs:")
    for v in vertices:
        pinned = " (PINNED)" if len(value_range[v]) == 1 else ""
        print(f"  {v}: {sorted(value_range[v])}{pinned}")
    print()

    return gamma_R, rdfs


# --------------------------------------------------------------------------- #
# Interactive command-line interface
# --------------------------------------------------------------------------- #

def _parse_vertex_list(raw):
    """Parse a comma/space separated list of vertex labels."""
    raw = raw.replace(",", " ")
    return [tok.strip() for tok in raw.split() if tok.strip()]


def _parse_edge_list(raw):
    """
    Parse edges given as e.g.  "a-b, b-c, c-d"  or  "a b, b c"  into a
    list of tuples.  Returns [] for empty input.
    """
    raw = raw.strip()
    if not raw:
        return []
    edges = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk and " " not in chunk:
            u, v = chunk.split("-")
        else:
            u, v = chunk.split()
        edges.append((u.strip(), v.strip()))
    return edges


def run_interactive():
    print("=" * 70)
    print(" Minimum Roman Dominating Functions of a Signed Graph")
    print("=" * 70)
    print()
    print("You will be asked for:")
    print("  1) the number of vertices,")
    print("  2) the vertex labels (or press Enter to use 1..n),")
    print("  3) the positive edges,")
    print("  4) the negative edges.")
    print()
    print("Edges can be entered like:   a-b, b-c, c-d")
    print("or:                          a b, b c, c d")
    print()

    n = int(input("Number of vertices: ").strip())

    label_raw = input(
        f"Vertex labels (comma/space separated, or Enter for 1..{n}): "
    ).strip()
    if label_raw:
        vertices = _parse_vertex_list(label_raw)
        if len(vertices) != n:
            raise ValueError(
                f"You said {n} vertices but gave {len(vertices)} labels."
            )
    else:
        vertices = [str(i) for i in range(1, n + 1)]

    pos_raw = input("Positive edges (e.g. a-b, b-c): ")
    pos_edges = _parse_edge_list(pos_raw)

    neg_raw = input("Negative edges (e.g. c-d): ")
    neg_edges = _parse_edge_list(neg_raw)

    if n > 16:
        print(
            f"\nWarning: brute force checks 3^{n} = {3**n:,} functions; "
            "this may take a while.\n"
        )

    summarize(vertices, pos_edges, neg_edges)


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    run_interactive()