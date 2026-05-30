#!/usr/bin/env python3
"""Solve the open Rural-Postman / Chinese-Postman optimum for one component
to verify whether the beam-search plan is structurally optimal.

Approach (exact for small components):
  1. Build the component's multi-graph G with required edges.
  2. base_cost = sum of all edge lengths (every edge traversed once).
  3. Identify odd-degree nodes O.
  4. For each candidate (start, end) pair (start, t):
       Effective odd set after fixing start, t as path endpoints:
           eff_odd = O Δ {start, t}
       Min-weight perfect matching on eff_odd (pairs to be deadheaded).
       Deadhead cost between u, v = min(
           penup_penalty * euclidean(u, v),     # pen-up shortcut
           graph_dijkstra_distance(u, v),       # walk along covered/uncovered edges
       )
       total = base_cost + sum(deadhead pairs)
  5. Pick min over all (start, t).
  6. Add entry-jump cost from `prev_pos` to `start`.

Compares the optimal (start, t, total) to the beam-search plan's choice.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize

import sys
sys.path.insert(0, str(Path(__file__).parent))
from skeleton_plan import build_graph, prune_stubs, euclidean
from train_char_recognizer import parse_words_txt


def open_cpp_optimum(nodes, comp_edges, prev_pos, penup_penalty):
    """Returns (best_total, best_start, best_end, best_jump, best_deadhead)
    where best_total = best_jump + sum(edge_lengths) + best_deadhead.
    """
    G = nx.MultiGraph()
    for eid, e in comp_edges:
        G.add_edge(e["a"], e["b"], weight=e["length"], eid=eid)
    base = sum(e["length"] for _, e in comp_edges)

    # Graph shortest-path distance (free pen-down deadhead if path lies on graph).
    apsp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    def deadhead(u, v):
        if u == v:
            return 0.0
        graph_d = apsp[u].get(v, float("inf"))
        eucl = penup_penalty * np.hypot(nodes[u][0] - nodes[v][0],
                                        nodes[u][1] - nodes[v][1])
        return min(graph_d, eucl)

    odd = {n for n, d in G.degree() if d % 2 == 1}
    node_list = list(G.nodes())

    best_total = float("inf")
    best = None
    for start in node_list:
        entry_jump = penup_penalty * np.hypot(prev_pos[0] - nodes[start][0],
                                              prev_pos[1] - nodes[start][1])
        for end in node_list:
            if end == start:
                continue
            eff = odd ^ {start, end}
            if len(eff) % 2 != 0:
                continue
            if not eff:
                dead_cost = 0.0
            else:
                mg = nx.Graph()
                eff_list = list(eff)
                for i, u in enumerate(eff_list):
                    for v in eff_list[i + 1:]:
                        # max_weight_matching maximizes; negate to minimize.
                        mg.add_edge(u, v, weight=-deadhead(u, v))
                matching = nx.algorithms.matching.max_weight_matching(
                    mg, maxcardinality=True)
                if len(matching) * 2 < len(eff_list):
                    continue
                dead_cost = sum(-mg[u][v]["weight"] for u, v in matching)
            total = entry_jump + base + dead_cost
            if total < best_total:
                best_total = total
                best = {
                    "start": start, "end": end,
                    "entry_jump": entry_jump,
                    "base": base,
                    "deadhead": dead_cost,
                }
    return best_total, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word-id", default="c03-007-02-07")
    ap.add_argument("--words-txt", default="data/iam_words/iam_words/words.txt")
    ap.add_argument("--words-dir", default="data/iam_words/iam_words/words")
    ap.add_argument("--prev-pos", type=str, default="61,141",
                    help="Comma-separated y,x of pen position before entering "
                         "the analyzed component (default: end of 'a' from beam plan).")
    ap.add_argument("--component-idx", type=int, default=-1,
                    help="Index into the L→R component order to analyze (-1 = last = bby).")
    ap.add_argument("--penup-penalty", type=float, default=2.0)
    ap.add_argument("--stub-prune-threshold", type=float, default=3.0)
    args = ap.parse_args()

    records = parse_words_txt(Path(args.words_txt), words_dir=Path(args.words_dir))
    rec = next(r for r in records if r["word_id"] == args.word_id)
    p = Path(args.words_dir) / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
    arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    ink = gaussian_filter(arr, sigma=0.6) < 0.5
    skel = skeletonize(ink)
    nodes, edges, comp_of_edge, n_components = build_graph(skel)
    nodes, edges, comp_of_edge, n_components = prune_stubs(
        nodes, edges, comp_of_edge, n_components, args.stub_prune_threshold)
    print(f"graph: {len(edges)} edges  {n_components} components")

    comp_min_x = {c: min(p[1] for eid, e in enumerate(edges) if comp_of_edge[eid] == c
                          for p in e["points"]) for c in range(n_components)}
    components_lr = sorted(range(n_components), key=lambda c: comp_min_x[c])
    target_comp = components_lr[args.component_idx]
    comp_edges = [(eid, edges[eid]) for eid in range(len(edges))
                  if comp_of_edge[eid] == target_comp]
    print(f"analyzing component idx={args.component_idx} (id={target_comp}): "
          f"{len(comp_edges)} edges")

    prev_y, prev_x = map(int, args.prev_pos.split(","))
    prev_pos = (prev_y, prev_x)
    print(f"prev pen position: {prev_pos}  penup_penalty={args.penup_penalty}")

    best_total, best = open_cpp_optimum(nodes, comp_edges, prev_pos, args.penup_penalty)
    print()
    print("=== exact open RPP solution ===")
    print(f"  start node {best['start']} = {nodes[best['start']]}")
    print(f"  end   node {best['end']}   = {nodes[best['end']]}")
    print(f"  entry jump cost: {best['entry_jump']:.1f}  "
          f"(physical jump = {best['entry_jump'] / args.penup_penalty:.1f} px)")
    print(f"  base (skeleton sum): {best['base']:.1f}")
    print(f"  deadhead cost: {best['deadhead']:.1f}")
    print(f"  TOTAL weighted cost (this component): {best_total:.1f}")
    print()

    # For comparison: cost of the beam-plan choice (entry at y-tail-ish).
    # The beam picked start near (99, 215). Find the closest node.
    beam_target = (99, 215)
    closest = min(range(len(nodes)),
                  key=lambda i: np.hypot(nodes[i][0] - beam_target[0],
                                         nodes[i][1] - beam_target[1])
                  if i in {e["a"] for _, e in comp_edges} | {e["b"] for _, e in comp_edges}
                  else float("inf"))
    bj = args.penup_penalty * np.hypot(prev_pos[0] - nodes[closest][0],
                                        prev_pos[1] - nodes[closest][1])
    print(f"=== beam's chosen entry (~y-tail) ===")
    print(f"  closest node {closest} = {nodes[closest]}")
    print(f"  entry jump cost: {bj:.1f}  (physical: {bj/args.penup_penalty:.1f} px)")


if __name__ == "__main__":
    main()
