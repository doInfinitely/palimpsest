#!/usr/bin/env python3
"""Plan a stroke trajectory that traces an IAM word's skeleton.

Pipeline:
  1. Skeletonize the word.
  2. Build a graph from the skeleton: nodes = endpoints + junctions,
     edges = paths of regular (degree-2) pixels between them.
  3. Group edges into connected components, ordered left → right.
  4. Beam search over (pen_pos, covered_edges, total_distance):
       * From a node, branch on uncovered outgoing edges.
       * After a dead-end, "pen up" → jump to a sampled point on an
         uncovered edge (cost = euclidean distance), then "pen down".
       * Initial pen-up position sampled to the LEFT of the word (try a
         few seeds).
       * Score = covered_length / total_distance; expand top-K.
  5. Render: faded word + trajectory (color gradient along time),
     hollow dots at pen-up moments, solid dots at pen-down. Trajectory
     ends with a pen-up.

Usage:
    python3 skeleton_plan.py \\
        --word-id c03-007-02-07 \\
        --words-txt data/iam_words/iam_words/words.txt \\
        --words-dir data/iam_words/iam_words/words \\
        --out eval_output/plan_shabby.png
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label as cc_label
from skimage.morphology import skeletonize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx

from train_char_recognizer import parse_words_txt


# ------------------------------------------------------------------
# Skeleton + graph
# ------------------------------------------------------------------

def neighbors_8(p: Tuple[int, int], skel_set) -> List[Tuple[int, int]]:
    y, x = p
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            q = (y + dy, x + dx)
            if q in skel_set:
                out.append(q)
    return out


def build_graph(skel: np.ndarray):
    """Returns (nodes, edges, comp_of_edge, n_components).

    nodes: list[(y, x)]                # endpoints + junctions
    edges: list[{a, b, points, length}]  # path between two node ids
    comp_of_edge: list[int]            # component id per edge
    """
    skel_pixels = [tuple(p) for p in np.argwhere(skel)]
    skel_set = set(skel_pixels)
    nbr_map = {p: neighbors_8(p, skel_set) for p in skel_pixels}

    node_pixels = [p for p in skel_pixels if len(nbr_map[p]) != 2]
    node_set = set(node_pixels)
    px_to_node = {p: i for i, p in enumerate(node_pixels)}

    edges = []
    edge_visited: set = set()  # set of frozenset({px_a, px_b}) of FIRST step

    def walk(start_node: Tuple[int, int], first_step: Tuple[int, int]):
        """Walk from start_node through first_step until another node."""
        key = frozenset([start_node, first_step])
        if key in edge_visited:
            return None
        edge_visited.add(key)
        path = [start_node, first_step]
        prev, curr = start_node, first_step
        while curr not in node_set:
            nxt_choices = [q for q in nbr_map[curr] if q != prev]
            if not nxt_choices:
                # Dead-end mid-walk (shouldn't happen for valid skeleton)
                break
            nxt = nxt_choices[0]
            kk = frozenset([curr, nxt])
            if kk in edge_visited:
                break
            edge_visited.add(kk)
            path.append(nxt)
            prev, curr = curr, nxt
        if curr not in px_to_node:
            return None
        return {
            "a": px_to_node[start_node],
            "b": px_to_node[curr],
            "points": path,
            "length": float(sum(
                np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                for i in range(len(path) - 1))),
        }

    for npx in node_pixels:
        for step in nbr_map[npx]:
            e = walk(npx, step)
            if e is not None:
                edges.append(e)

    # Closed loops (no nodes) — find unvisited skeleton pixels and treat
    # them as standalone loops. For simplicity, pick any pixel on the
    # loop as a 1-node graph with a single self-edge.
    visited_loop_px = set()
    for k in edge_visited:
        for p in k:
            visited_loop_px.add(p)
    leftover = [p for p in skel_pixels if p not in visited_loop_px and len(nbr_map[p]) == 2]
    while leftover:
        seed = leftover.pop()
        if seed in visited_loop_px:
            continue
        node_idx = len(node_pixels)
        node_pixels.append(seed)
        px_to_node[seed] = node_idx
        # Walk one direction around the loop until back to seed.
        path = [seed]
        prev, curr = seed, nbr_map[seed][0]
        path.append(curr)
        visited_loop_px.add(seed)
        visited_loop_px.add(curr)
        while curr != seed:
            nxts = [q for q in nbr_map[curr] if q != prev]
            if not nxts:
                break
            nxt = nxts[0]
            path.append(nxt)
            visited_loop_px.add(nxt)
            prev, curr = curr, nxt
            if len(path) > len(skel_pixels):
                break
        edges.append({
            "a": node_idx, "b": node_idx, "points": path,
            "length": float(sum(
                np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                for i in range(len(path) - 1))),
        })
        leftover = [p for p in skel_pixels if p not in visited_loop_px and len(nbr_map[p]) == 2]

    # Connected components via union-find on nodes joined by edges.
    parent = list(range(len(node_pixels)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for e in edges:
        union(e["a"], e["b"])
    comp_root_to_id = {}
    comp_of_node = []
    for n_idx in range(len(node_pixels)):
        r = find(n_idx)
        if r not in comp_root_to_id:
            comp_root_to_id[r] = len(comp_root_to_id)
        comp_of_node.append(comp_root_to_id[r])
    comp_of_edge = [comp_of_node[e["a"]] for e in edges]
    n_components = len(comp_root_to_id)

    return node_pixels, edges, comp_of_edge, n_components


# ------------------------------------------------------------------
# Beam search
# ------------------------------------------------------------------

class State:
    __slots__ = ("pos", "covered", "dist", "log", "comp_done")
    def __init__(self, pos, covered: FrozenSet[int], dist: float,
                 log: tuple, comp_done: FrozenSet[int]):
        self.pos = pos          # (y, x)
        self.covered = covered  # frozenset of edge ids covered
        self.dist = dist
        self.log = log          # tuple of action records
        self.comp_done = comp_done

    def score(self, total_edge_length: float) -> float:
        cov_len = sum(EDGES[i]["length"] for i in self.covered)
        # higher is better: coverage minus α·distance
        return cov_len - 0.0 * self.dist  # coverage primary; tie-break by distance

    def key(self):
        # for dedup: same pos + same covered set
        return (self.pos, self.covered)


def euclidean(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def expand(state: State, edges, node_pixels, comp_of_edge, components_lr,
           sample_targets_per_edge: int = 4,
           penup_penalty: float = 1.0) -> List[State]:
    """Generate successor states.

    From state.pos, two kinds of moves:
      (A) If pos is a node, walk an uncovered edge incident to that node.
      (B) Pen up + jump to a sampled point on an uncovered edge in the
          current target component (or next one if current is fully
          covered), then pen down at that sampled point.
    """
    succs = []
    pos = state.pos
    covered = state.covered

    # Determine current target component: the leftmost component with
    # uncovered edges.
    target_comp = None
    for c in components_lr:
        if any((eid not in covered) for eid in range(len(edges))
               if comp_of_edge[eid] == c):
            target_comp = c
            break
    if target_comp is None:
        return []  # all covered

    # (A) walk an uncovered outgoing edge from pos if pos is a node.
    pos_node_idx = NODE_PX_TO_IDX.get(pos)
    if pos_node_idx is not None:
        for eid, e in enumerate(edges):
            if eid in covered:
                continue
            if e["a"] != pos_node_idx and e["b"] != pos_node_idx:
                continue
            if comp_of_edge[eid] != target_comp:
                continue
            # Walk along the edge from pos to the other node.
            if e["a"] == pos_node_idx:
                pts = e["points"]
                end_pos = node_pixels[e["b"]]
            else:
                pts = list(reversed(e["points"]))
                end_pos = node_pixels[e["a"]]
            new_state = State(
                pos=end_pos,
                covered=covered | {eid},
                dist=state.dist + e["length"],
                log=state.log + (("draw", eid, pts),),
                comp_done=state.comp_done,
            )
            succs.append(new_state)

    # (B) Retrace a COVERED edge to another node (free pen-down travel).
    # Useful when there's no uncovered out-edge here but a nearby node has one.
    if pos_node_idx is not None:
        for eid, e in enumerate(edges):
            if eid not in covered:
                continue
            if e["a"] != pos_node_idx and e["b"] != pos_node_idx:
                continue
            if e["a"] == pos_node_idx:
                end_pos = node_pixels[e["b"]]
            else:
                end_pos = node_pixels[e["a"]]
            if end_pos == pos:
                continue
            new_state = State(
                pos=end_pos,
                covered=covered,
                dist=state.dist + e["length"],
                log=state.log + (("retrace", eid),),
                comp_done=state.comp_done,
            )
            succs.append(new_state)

    # (C) pen-up jump to a sampled point on an uncovered edge of target_comp.
    # Always offered; the beam picks based on cost.
    candidates = []
    for eid, e in enumerate(edges):
        if eid in covered or comp_of_edge[eid] != target_comp:
            continue
        pts = e["points"]
        S = max(1, sample_targets_per_edge)
        idxs = np.linspace(0, len(pts) - 1, S).astype(int)
        for ii in idxs:
            candidates.append((eid, ii, pts[ii]))
    for eid, ii, p in candidates:
        jump_dist = euclidean(pos, p)
        if jump_dist < 0.5:
            continue  # already there
        new_state = State(
            pos=p,
            covered=covered,
            dist=state.dist + jump_dist * penup_penalty,
            log=state.log + (("jump", p),),
            comp_done=state.comp_done,
        )
        succs.append(new_state)

    return succs


def beam_search(edges, node_pixels, comp_of_edge, components_lr,
                seed_positions: List[Tuple[int, int]],
                beam_width: int = 64, max_iters: int = 5000,
                sample_targets_per_edge: int = 2,
                penup_penalty: float = 1.0):
    total_len = sum(e["length"] for e in edges)
    # Initial beam: each seed is a pen-up state at that position.
    beam = [
        State(pos=s, covered=frozenset(), dist=0.0, log=(("init", s),),
              comp_done=frozenset())
        for s in seed_positions
    ]
    best = None
    seen: Dict[Tuple, float] = {}

    for it in range(max_iters):
        if not beam:
            break
        # Find any state that has fully covered all edges
        for st in beam:
            if len(st.covered) == len(edges):
                if best is None or st.dist < best.dist:
                    best = st
        # Stop early if no new expansions possible (all beams fully covered)
        if best is not None and all(len(st.covered) == len(edges) for st in beam):
            break

        next_beam_pool = []
        for st in beam:
            if len(st.covered) == len(edges):
                next_beam_pool.append(st)  # carry forward
                continue
            for s2 in expand(st, edges, node_pixels, comp_of_edge, components_lr,
                             sample_targets_per_edge=sample_targets_per_edge,
                             penup_penalty=penup_penalty):
                k = s2.key()
                if k in seen and seen[k] <= s2.dist:
                    continue
                seen[k] = s2.dist
                next_beam_pool.append(s2)
        if not next_beam_pool:
            break
        # Prune beam: prefer high coverage, then low distance.
        next_beam_pool.sort(key=lambda s: (-len(s.covered),
                                           -sum(edges[i]["length"] for i in s.covered),
                                           s.dist))
        beam = next_beam_pool[:beam_width]

    if best is None:
        # Best partial: state that maximizes coverage / minimizes distance
        beam.sort(key=lambda s: (-len(s.covered),
                                 -sum(edges[i]["length"] for i in s.covered),
                                 s.dist))
        best = beam[0]
    return best, total_len


def seed_y_of(state: State) -> int:
    """Recover the y of the (init) seed that produced this state."""
    for action in state.log:
        if action[0] == "init":
            return int(action[1][0])
    return -1


def prune_stubs(nodes, edges, comp_of_edge, n_components,
                threshold: float = 3.0):
    """Two passes, iterated to fixed point:
       (1) Drop hairs: edges with length < threshold AND a degree-1 endpoint.
       (2) Contract junction-blob edges: edges with length < threshold whose
           BOTH endpoints have degree ≥ 3 (a tight cluster of junctions
           produced by skeletonize on thick stroke crossings)."""
    nodes = list(nodes)  # we may extend this via contraction (we don't here, but keep mutable)
    while True:
        degree = {i: 0 for i in range(len(nodes))}
        for e in edges:
            degree[e["a"]] += 1
            if e["a"] != e["b"]:
                degree[e["b"]] += 1

        # (1) drop hairs
        kept = []
        dropped = 0
        for e in edges:
            if e["length"] < threshold and (degree[e["a"]] == 1 or degree[e["b"]] == 1):
                dropped += 1
                continue
            kept.append(e)
        edges = kept

        # (2) contract short junction-junction edges using union-find
        parent = list(range(len(nodes)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        contracted = 0
        recompute = True
        # rebuild degrees after the hair drop above
        degree = {i: 0 for i in range(len(nodes))}
        for e in edges:
            degree[e["a"]] += 1
            if e["a"] != e["b"]:
                degree[e["b"]] += 1
        for e in edges:
            if e["a"] == e["b"]:
                continue
            if e["length"] < threshold and degree[e["a"]] >= 3 and degree[e["b"]] >= 3:
                ra, rb = find(e["a"]), find(e["b"])
                if ra != rb:
                    parent[ra] = rb
                    contracted += 1
        if contracted > 0:
            # Remap edges to representative nodes; drop self-loops produced by contraction.
            new_edges = []
            for e in edges:
                ra, rb = find(e["a"]), find(e["b"])
                if ra == rb and e["length"] < threshold:
                    continue  # contracted away
                new_edges.append({**e, "a": ra, "b": rb})
            edges = new_edges
        if dropped == 0 and contracted == 0:
            break

    # Drop now-isolated nodes (no edges touch them) — keep the original
    # node list indices stable (don't re-pack) so px_to_node mapping doesn't
    # need updating. Just rebuild components.
    parent = list(range(len(nodes)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in edges:
        ra, rb = find(e["a"]), find(e["b"])
        if ra != rb:
            parent[ra] = rb
    # Map nodes that are touched by edges into a tighter component id space.
    touched = set()
    for e in edges:
        touched.add(e["a"])
        touched.add(e["b"])
    comp_root_to_id = {}
    for n_idx in touched:
        r = find(n_idx)
        if r not in comp_root_to_id:
            comp_root_to_id[r] = len(comp_root_to_id)
    comp_of_edge = [comp_root_to_id[find(e["a"])] for e in edges]
    return nodes, edges, comp_of_edge, len(comp_root_to_id)


# ------------------------------------------------------------------
# Exact open-RPP solver (per-component)
# ------------------------------------------------------------------

def solve_component_exact(nodes, comp_edges, prev_pos, penup_penalty):
    """Optimal open Rural Postman traversal of one connected component.

    Returns (total_cost_for_component, action_log, end_pos), where action_log
    is a list of actions in the same vocabulary as the beam search:
      ('jump', target_pos), ('draw', eid, points), ('retrace', eid)

    The first action is a 'jump' from prev_pos to the chosen start node (omitted
    if already there).
    """
    G = nx.MultiGraph()
    eid_to_edge = dict(comp_edges)
    for eid, e in comp_edges:
        G.add_edge(e["a"], e["b"], weight=e["length"], kind="real", eid=eid)
    base = sum(e["length"] for _, e in comp_edges)

    apsp_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    apsp_path = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))

    def dh(u, v):
        if u == v:
            return ("nothing", 0.0, [u])
        eucl = penup_penalty * float(np.hypot(nodes[u][0] - nodes[v][0],
                                              nodes[u][1] - nodes[v][1]))
        gd = apsp_dist[u].get(v, float("inf"))
        gpath = apsp_path[u].get(v)
        if gd <= eucl:
            return ("walk", gd, gpath)
        return ("jump", eucl, [u, v])

    odd = {n for n, d in G.degree() if d % 2 == 1}
    nlist = list(G.nodes())

    best = None
    # Degenerate: 1-node component (only self-loops). Allow start == end.
    if len(nlist) == 1:
        start = end = nlist[0]
        entry = penup_penalty * float(np.hypot(prev_pos[0] - nodes[start][0],
                                               prev_pos[1] - nodes[start][1]))
        best = {"total": entry + base, "start": start, "end": end,
                "pairs": [], "entry": entry}
    for start in nlist:
        entry = penup_penalty * float(np.hypot(prev_pos[0] - nodes[start][0],
                                               prev_pos[1] - nodes[start][1]))
        for end in nlist:
            if end == start:
                continue
            eff = odd ^ {start, end}
            if len(eff) % 2 != 0:
                continue
            if not eff:
                pairs, dead = [], 0.0
            else:
                mg = nx.Graph()
                el = list(eff)
                for i, u in enumerate(el):
                    for v in el[i + 1:]:
                        _, c, _ = dh(u, v)
                        mg.add_edge(u, v, weight=-c)
                m = nx.algorithms.matching.max_weight_matching(mg, maxcardinality=True)
                if len(m) * 2 < len(el):
                    continue
                pairs = list(m)
                dead = sum(-mg[u][v]["weight"] for u, v in pairs)
            tot = entry + base + dead
            if best is None or tot < best["total"]:
                best = {"total": tot, "start": start, "end": end,
                        "pairs": pairs, "entry": entry}

    # Reconstruct Euler path on augmented multigraph.
    AG = nx.MultiGraph()
    for u, v, k, d in G.edges(keys=True, data=True):
        AG.add_edge(u, v, key=("real", k), kind="real", eid=d["eid"], weight=d["weight"])
    next_dup_id = 0
    for u, v in best["pairs"]:
        kind, _, path = dh(u, v)
        if kind == "walk":
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                base_eid = None
                for k, dat in G[a][b].items():
                    base_eid = dat["eid"]
                    base_w = dat["weight"]
                    break
                AG.add_edge(a, b, key=("dup", next_dup_id), kind="walk-dup",
                            eid=base_eid, weight=base_w)
                next_dup_id += 1
        else:  # jump
            jw = penup_penalty * float(np.hypot(nodes[u][0] - nodes[v][0],
                                                nodes[u][1] - nodes[v][1]))
            AG.add_edge(u, v, key=("penup", next_dup_id), kind="penup", weight=jw)
            next_dup_id += 1

    # Euler path from start to end.
    log = []
    if best["entry"] > 0.5:
        log.append(("jump", nodes[best["start"]]))
    covered = set()
    for u, v, key in nx.eulerian_path(AG, source=best["start"], keys=True):
        edata = AG.edges[u, v, key]
        if edata["kind"] == "penup":
            log.append(("jump", nodes[v]))
        else:
            eid = edata["eid"]
            e = eid_to_edge[eid]
            if eid not in covered:
                pts = e["points"] if e["a"] == u else list(reversed(e["points"]))
                log.append(("draw", eid, pts))
                covered.add(eid)
            else:
                log.append(("retrace", eid))

    return best["total"], log, nodes[best["end"]]


def plan_exact(nodes, edges, comp_of_edge, components_lr, seed_positions,
               penup_penalty: float):
    """Process components L→R, solving each one exactly. Try each seed and
    pick the one whose total cost across all components is smallest."""
    best_total = float("inf")
    best_log = None
    best_seed = None
    for seed in seed_positions:
        total = 0.0
        log = [("init", seed)]
        cur = seed
        for c in components_lr:
            comp_edges = [(eid, edges[eid]) for eid in range(len(edges))
                          if comp_of_edge[eid] == c]
            cost, comp_log, end_pos = solve_component_exact(
                nodes, comp_edges, cur, penup_penalty,
            )
            total += cost
            log.extend(comp_log)
            cur = end_pos
        if total < best_total:
            best_total = total
            best_log = log
            best_seed = seed

    # Build a State so the rest of the pipeline can use it uniformly.
    all_edge_ids = set(range(len(edges)))
    s = State(pos=cur if best_log is None else None,  # filled below
              covered=frozenset(all_edge_ids),
              dist=best_total,
              log=tuple(best_log),
              comp_done=frozenset())
    return s, best_seed


# ------------------------------------------------------------------
# Render
# ------------------------------------------------------------------

def trajectory_segments(state: State) -> Tuple[List[List[Tuple[float, float]]],
                                                List[Tuple[Tuple[float, float], str]]]:
    """Returns:
        segments: list of (segment_pts, kind) where kind in {"draw", "jump"}
        markers: list of (pos, kind) where kind in {"penup", "pendown"}
    Trajectory always ends with a pen-up.
    """
    segments = []
    markers = []
    pen_down = False
    cur_pos = None
    for action in state.log:
        if action[0] == "init":
            cur_pos = action[1]
            markers.append((cur_pos, "penup"))  # start with pen up
        elif action[0] == "jump":
            target = action[1]
            if pen_down:
                # implicit pen-up before the jump
                markers.append((cur_pos, "penup"))
                pen_down = False
            segments.append(([cur_pos, target], "jump"))
            cur_pos = target
            markers.append((cur_pos, "pendown"))
            pen_down = True
        elif action[0] == "draw":
            eid, pts = action[1], action[2]
            if not pen_down:
                if cur_pos is not None and cur_pos != pts[0]:
                    segments.append(([cur_pos, pts[0]], "jump"))
                    markers.append((pts[0], "pendown"))
                cur_pos = pts[0]
                pen_down = True
            segments.append((list(pts), "draw"))
            cur_pos = pts[-1]
        elif action[0] == "retrace":
            eid = action[1]
            # Walk along the covered edge from cur_pos to its other endpoint.
            e = EDGES[eid]
            pa, pb = NODE_TO_PX_LOCAL[e["a"]], NODE_TO_PX_LOCAL[e["b"]]
            if cur_pos == pa:
                pts = e["points"]
            else:
                pts = list(reversed(e["points"]))
            if not pen_down:
                cur_pos = pts[0]
                markers.append((cur_pos, "pendown"))
                pen_down = True
            segments.append((list(pts), "retrace"))
            cur_pos = pts[-1]
    # End with pen up
    if pen_down:
        markers.append((cur_pos, "penup"))
    return segments, markers


def render(arr: np.ndarray, segments, markers, out_path: Path):
    H, W = arr.shape
    fig, ax = plt.subplots(figsize=(W / 50, H / 50), dpi=200)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1, alpha=0.45)

    # Time-color gradient over all segments.
    line_segments = []
    seg_kinds = []
    for pts, kind in segments:
        for i in range(len(pts) - 1):
            (y0, x0) = pts[i]
            (y1, x1) = pts[i + 1]
            line_segments.append([(x0, y0), (x1, y1)])
            seg_kinds.append(kind)
    if line_segments:
        n = len(line_segments)
        cmap = plt.colormaps["plasma"]
        colors = []
        for i, k in enumerate(seg_kinds):
            base = cmap(i / max(n - 1, 1))
            if k == "jump":
                # render jumps with a translucent dashed-feel (lighter alpha)
                colors.append((base[0], base[1], base[2], 0.35))
            else:
                colors.append((base[0], base[1], base[2], 0.95))
        lc = LineCollection(line_segments, colors=colors, linewidths=1.4)
        ax.add_collection(lc)

    # Markers.
    for (y, x), kind in markers:
        if kind == "penup":
            ax.plot(x, y, marker="o", markerfacecolor="none",
                    markeredgecolor="black", markersize=6, markeredgewidth=1.0)
        else:
            ax.plot(x, y, marker="o", markerfacecolor="black",
                    markeredgecolor="black", markersize=4)

    ax.set_xlim(-2, W + 2); ax.set_ylim(H + 2, -2)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

NODE_PX_TO_IDX: Dict[Tuple[int, int], int] = {}
NODE_TO_PX_LOCAL: List[Tuple[int, int]] = []
EDGES: List[dict] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word-id", required=True)
    ap.add_argument("--words-txt", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ink-threshold", type=float, default=0.5)
    ap.add_argument("--blur-sigma", type=float, default=0.6)
    ap.add_argument("--n-seeds", type=int, default=4,
                    help="Number of pen-up start positions left of the word.")
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--samples-per-edge", type=int, default=2)
    ap.add_argument("--refine-seeds", action="store_true", default=True,
                    help="Coarse-to-fine seed search: after the first pass, "
                         "resample seeds in a tight window around the winner.")
    ap.add_argument("--no-refine-seeds", dest="refine_seeds", action="store_false")
    ap.add_argument("--refine-window", type=int, default=15,
                    help="Half-window (in pixels of y) around the best coarse "
                         "seed to resample at refinement time.")
    ap.add_argument("--n-refine-seeds", type=int, default=6)
    ap.add_argument("--stub-prune-threshold", type=float, default=3.0,
                    help="Drop skeleton edges shorter than this with a "
                         "degree-1 endpoint (skeletonization hairs). 0 disables.")
    ap.add_argument("--penup-penalty", type=float, default=2.0,
                    help="Cost multiplier for pen-up jump distance. The beam's "
                         "reported 'distance' becomes weighted: ink_distance + "
                         "penup_penalty * jump_distance.")
    ap.add_argument("--exact-component", action="store_true",
                    help="Replace beam search with exact open-RPP solve per "
                         "connected component (Edmonds matching + Euler path).")
    args = ap.parse_args()

    records = parse_words_txt(Path(args.words_txt), words_dir=Path(args.words_dir))
    rec = next((r for r in records if r["word_id"] == args.word_id), None)
    if rec is None:
        raise SystemExit(f"No record for word_id={args.word_id}")
    p = Path(args.words_dir) / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
    img = Image.open(p).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr_blur = gaussian_filter(arr, sigma=args.blur_sigma) if args.blur_sigma > 0 else arr
    ink = arr_blur < args.ink_threshold
    skel = skeletonize(ink)
    H, W = arr.shape
    print(f"word: {rec['text']!r}  image {arr.shape}  ink {int(ink.sum())}  skel {int(skel.sum())}")

    nodes, edges, comp_of_edge, n_components = build_graph(skel)
    print(f"graph (raw): {len(nodes)} nodes  {len(edges)} edges  {n_components} components")
    if args.stub_prune_threshold > 0:
        nodes, edges, comp_of_edge, n_components = prune_stubs(
            nodes, edges, comp_of_edge, n_components,
            threshold=args.stub_prune_threshold,
        )
        print(f"graph (pruned): {len(edges)} edges  {n_components} components")
    global NODE_PX_TO_IDX, NODE_TO_PX_LOCAL, EDGES
    NODE_PX_TO_IDX = {p: i for i, p in enumerate(nodes)}
    NODE_TO_PX_LOCAL = list(nodes)
    EDGES = edges

    # Order components left → right by leftmost edge x.
    comp_min_x = {c: min(p[1] for eid, e in enumerate(edges) if comp_of_edge[eid] == c
                          for p in e["points"]) for c in range(n_components)}
    components_lr = sorted(range(n_components), key=lambda c: comp_min_x[c])

    # Seed positions: a few choices left of the word, spanning vertical extent.
    word_left = min(p[1] for e in edges for p in e["points"])
    word_top = min(p[0] for e in edges for p in e["points"])
    word_bot = max(p[0] for e in edges for p in e["points"])
    seed_x = max(0, word_left - 20)
    ys = np.linspace(word_top, word_bot, args.n_seeds).astype(int)
    seeds = [(int(y), int(seed_x)) for y in ys]
    print(f"coarse seeds: {seeds}")

    if args.exact_component:
        best, chosen_seed = plan_exact(nodes, edges, comp_of_edge, components_lr,
                                       seeds, args.penup_penalty)
        cov_len = sum(e["length"] for e in edges)
        total_len = cov_len
        print(f"exact plan: covered all {len(edges)} edges, "
              f"weighted cost {best.dist:.1f}, seed={chosen_seed}")
    else:
        best, total_len = beam_search(edges, nodes, comp_of_edge, components_lr,
                                      seeds, beam_width=args.beam_width,
                                      sample_targets_per_edge=args.samples_per_edge,
                                      penup_penalty=args.penup_penalty)
        cov_len = sum(edges[i]["length"] for i in best.covered)
        print(f"coarse plan: covered {cov_len:.1f}/{total_len:.1f}  "
              f"distance {best.dist:.1f}  best seed y={seed_y_of(best)}")

    if args.refine_seeds and not args.exact_component:
        best_y = seed_y_of(best)
        y_lo = max(0, best_y - args.refine_window)
        y_hi = min(arr.shape[0] - 1, best_y + args.refine_window)
        refine_ys = np.linspace(y_lo, y_hi, args.n_refine_seeds).astype(int)
        refine_seeds = [(int(y), int(seed_x)) for y in refine_ys]
        print(f"refine seeds (around y={best_y}): {refine_seeds}")
        best_r, _ = beam_search(edges, nodes, comp_of_edge, components_lr,
                                refine_seeds, beam_width=args.beam_width,
                                sample_targets_per_edge=args.samples_per_edge,
                                penup_penalty=args.penup_penalty)
        cov_r = sum(edges[i]["length"] for i in best_r.covered)
        print(f"refine plan: covered {cov_r:.1f}/{total_len:.1f}  distance {best_r.dist:.1f}")
        if cov_r >= cov_len and best_r.dist < best.dist:
            print(f"  → refinement wins (Δdist {best.dist - best_r.dist:.1f})")
            best = best_r
            cov_len = cov_r
        else:
            print(f"  → coarse already best (kept)")
    # Decompose final plan into ink (draw), retrace (pen-down deadhead), and
    # jump (pen-up) physical distance.
    ink_d = jump_d = retrace_d = 0.0
    cur = None
    for a in best.log:
        if a[0] == "init":
            cur = a[1]
        elif a[0] == "jump":
            jump_d += euclidean(cur, a[1])
            cur = a[1]
        elif a[0] == "draw":
            pts = a[2]
            if cur is not None and cur != pts[0]:
                jump_d += euclidean(cur, pts[0])
            ink_d += sum(np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                         for i in range(len(pts)-1))
            cur = pts[-1]
        elif a[0] == "retrace":
            eid = a[1]
            e = edges[eid]
            pa = nodes[e["a"]]; pb = nodes[e["b"]]
            pts = e["points"] if cur == pa else list(reversed(e["points"]))
            retrace_d += sum(np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                             for i in range(len(pts)-1))
            cur = pts[-1]
    print(f"final plan: covered {cov_len:.1f}/{total_len:.1f} px  "
          f"({len(best.covered)}/{len(edges)} edges)  "
          f"weighted cost {best.dist:.1f}  "
          f"physical: ink={ink_d:.1f} retrace={retrace_d:.1f} jump={jump_d:.1f} "
          f"(penup_penalty={args.penup_penalty})")

    segments, markers = trajectory_segments(best)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(arr, segments, markers, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
