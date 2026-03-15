"""
Stroke extraction, routing, and pseudo-online event generation.

Implements the palimpsest spec for:
  - Skeleton-based stroke decomposition from binarized glyphs
  - Word-level stroke routing via Held-Karp DP (small) or beam search (large)
  - Conversion to unified pen event format (dx, dy, pen_down, stroke_end, ...)
  - Arc-length resampling of polylines
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
OrientedStroke = Tuple[int, int]  # (stroke_id, dir), dir=0 fwd, 1 rev


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveStroke:
    stroke_id: int
    char_id: str
    char_index: int
    points_fwd: List[Point]
    arc_len: float
    x_center: float = 0.0

    @property
    def points_rev(self) -> List[Point]:
        return list(reversed(self.points_fwd))

    def points(self, direction: int) -> List[Point]:
        return self.points_fwd if direction == 0 else self.points_rev

    def start(self, direction: int) -> Point:
        return self.points(direction)[0]

    def end(self, direction: int) -> Point:
        return self.points(direction)[-1]


@dataclass
class StrokeRoute:
    ordered: List[Tuple[int, int]]  # [(stroke_idx, dir), ...]
    total_cost: float
    start_point: Point
    end_point: Point


@dataclass
class RouteWeights:
    lambda_down: float = 0.1
    lambda_up: float = 1.0
    lambda_curv: float = 0.0
    lambda_switch: float = 0.3
    lambda_back: float = 0.1
    lambda_start: float = 0.0
    lambda_end: float = 0.0


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------

def extract_strokes_from_binary(
    binary: np.ndarray,
    char_id: str,
    char_index: int,
    char_bbox_in_word: Tuple[float, float, float, float],
    word_size: Tuple[float, float],
    min_stroke_len: int = 3,
) -> List[PrimitiveStroke]:
    """Extract primitive strokes from a binary character image.

    Uses skeletonization + graph decomposition into maximal paths between
    endpoints/junctions.
    """
    from skimage.morphology import skeletonize

    if not binary.any():
        return []

    skel = skeletonize(binary)
    points_rc = np.argwhere(skel)
    if len(points_rc) < 2:
        return []

    points_set = set(map(tuple, points_rc))
    ww, wh = max(word_size[0], 1), max(word_size[1], 1)
    cx1, cy1, cx2, cy2 = char_bbox_in_word

    def _neighbors(r, c):
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in points_set:
                    out.append((r + dr, c + dc))
        return out

    # Classify pixels
    degree = {}
    for r, c in points_rc:
        degree[(r, c)] = len(_neighbors(r, c))

    endpoints = [p for p, d in degree.items() if d == 1]
    junctions = [p for p, d in degree.items() if d >= 3]
    special = set(endpoints) | set(junctions)

    visited_edges = set()
    strokes = []
    stroke_counter = 0

    def _trace_from(start, prev=None):
        path = [start]
        current = start
        while True:
            nbrs = [n for n in _neighbors(*current)
                     if n != prev and (current, n) not in visited_edges
                     and (n, current) not in visited_edges]
            if not nbrs:
                break
            # At junctions/endpoints, stop
            nxt = nbrs[0]
            visited_edges.add((current, nxt))
            visited_edges.add((nxt, current))
            path.append(nxt)
            if nxt in special and nxt != start:
                break
            prev = current
            current = nxt
        return path

    # Trace from endpoints first, then junctions, then any remaining
    for start in endpoints + junctions:
        for nbr in _neighbors(*start):
            if (start, nbr) in visited_edges:
                continue
            path = _trace_from(start)
            if len(path) >= min_stroke_len:
                # Normalize to word coordinates [0,1]
                pts_norm = []
                for r, c in path:
                    x = (cx1 + c) / ww
                    y = (cy1 + r) / wh
                    pts_norm.append((round(x, 6), round(y, 6)))

                arc = _polyline_arc_length(pts_norm)
                x_ctr = sum(p[0] for p in pts_norm) / len(pts_norm)

                strokes.append(PrimitiveStroke(
                    stroke_id=stroke_counter,
                    char_id=char_id,
                    char_index=char_index,
                    points_fwd=pts_norm,
                    arc_len=arc,
                    x_center=x_ctr,
                ))
                stroke_counter += 1

    # Catch any unvisited connected segments
    remaining = points_set - {p for e in visited_edges for p in e}
    if remaining:
        rem_list = sorted(remaining)
        visited_rem = set()
        for start in rem_list:
            if start in visited_rem:
                continue
            path = [start]
            visited_rem.add(start)
            current = start
            while True:
                nbrs = [n for n in _neighbors(*current)
                         if n not in visited_rem and n in remaining]
                if not nbrs:
                    break
                current = nbrs[0]
                visited_rem.add(current)
                path.append(current)

            if len(path) >= min_stroke_len:
                pts_norm = []
                for r, c in path:
                    x = (cx1 + c) / ww
                    y = (cy1 + r) / wh
                    pts_norm.append((round(x, 6), round(y, 6)))

                arc = _polyline_arc_length(pts_norm)
                x_ctr = sum(p[0] for p in pts_norm) / len(pts_norm)

                strokes.append(PrimitiveStroke(
                    stroke_id=stroke_counter,
                    char_id=char_id,
                    char_index=char_index,
                    points_fwd=pts_norm,
                    arc_len=arc,
                    x_center=x_ctr,
                ))
                stroke_counter += 1

    return strokes


# ---------------------------------------------------------------------------
# Polyline utilities
# ---------------------------------------------------------------------------

def _polyline_arc_length(pts: Sequence[Point]) -> float:
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
    return total


def resample_polyline(points: Sequence[Point], step: float) -> List[Point]:
    """Resample a polyline by arc length with given step size."""
    if len(points) <= 1:
        return list(points)

    out = [points[0]]
    carry = 0.0
    prev = points[0]

    for cur in points[1:]:
        seg_len = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if seg_len < 1e-12:
            prev = cur
            continue

        dx = (cur[0] - prev[0]) / seg_len
        dy = (cur[1] - prev[1]) / seg_len
        remaining = seg_len
        cp = prev

        while carry + remaining >= step:
            needed = step - carry
            new_pt = (cp[0] + dx * needed, cp[1] + dy * needed)
            out.append(new_pt)
            cp = new_pt
            remaining -= needed
            carry = 0.0

        carry += remaining
        prev = cur

    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


# ---------------------------------------------------------------------------
# Word-level stroke routing
# ---------------------------------------------------------------------------

def _internal_cost(s: PrimitiveStroke, d: int, w: RouteWeights) -> float:
    return w.lambda_down * s.arc_len


def _bridge_cost(
    sa: PrimitiveStroke, da: int,
    sb: PrimitiveStroke, db: int,
    w: RouteWeights,
) -> float:
    dist = math.hypot(
        sa.end(da)[0] - sb.start(db)[0],
        sa.end(da)[1] - sb.start(db)[1],
    )
    switch = w.lambda_switch if sa.char_id != sb.char_id else 0.0
    backward = w.lambda_back * max(0.0, sa.x_center - sb.x_center)
    return w.lambda_up * dist + switch + backward


def route_word_strokes_dp(
    strokes: List[PrimitiveStroke],
    weights: RouteWeights,
) -> StrokeRoute:
    """Exact Held-Karp DP for word-level stroke routing.

    Practical for m <= 16 strokes. For larger, use beam search.
    """
    m = len(strokes)
    if m == 0:
        return StrokeRoute([], 0.0, (0, 0), (0, 0))
    if m == 1:
        s = strokes[0]
        cost = _internal_cost(s, 0, weights)
        return StrokeRoute([(0, 0)], cost, s.start(0), s.end(0))

    INF = float("inf")
    # dp[(mask, i, d)] = best cost
    dp = {}
    parent = {}

    for i in range(m):
        for d in (0, 1):
            mask = 1 << i
            key = (mask, i, d)
            dp[key] = _internal_cost(strokes[i], d, weights)
            parent[key] = None

    for mask in range(1, 1 << m):
        for i in range(m):
            if not (mask & (1 << i)):
                continue
            for d1 in (0, 1):
                key = (mask, i, d1)
                if key not in dp:
                    continue
                for j in range(m):
                    if mask & (1 << j):
                        continue
                    for d2 in (0, 1):
                        new_mask = mask | (1 << j)
                        cand = (
                            dp[key]
                            + _bridge_cost(strokes[i], d1, strokes[j], d2, weights)
                            + _internal_cost(strokes[j], d2, weights)
                        )
                        new_key = (new_mask, j, d2)
                        if cand < dp.get(new_key, INF):
                            dp[new_key] = cand
                            parent[new_key] = key

    full = (1 << m) - 1
    best_key = None
    best_cost = INF
    for i in range(m):
        for d in (0, 1):
            key = (full, i, d)
            if key in dp and dp[key] < best_cost:
                best_cost = dp[key]
                best_key = key

    if best_key is None:
        return StrokeRoute([], 0.0, (0, 0), (0, 0))

    ordered = []
    cur = best_key
    while cur is not None:
        _, i, d = cur
        ordered.append((i, d))
        cur = parent[cur]
    ordered.reverse()

    fi, fd = ordered[0]
    li, ld = ordered[-1]
    return StrokeRoute(
        ordered=ordered,
        total_cost=best_cost,
        start_point=strokes[fi].start(fd),
        end_point=strokes[li].end(ld),
    )


def route_word_strokes_beam(
    strokes: List[PrimitiveStroke],
    weights: RouteWeights,
    beam_size: int = 32,
) -> StrokeRoute:
    """Beam search for word-level stroke routing when DP is too expensive."""
    m = len(strokes)
    if m == 0:
        return StrokeRoute([], 0.0, (0, 0), (0, 0))

    # beam: list of (cost, mask, last_i, last_d, ordered)
    beam = []
    for i in range(m):
        for d in (0, 1):
            cost = _internal_cost(strokes[i], d, weights)
            beam.append((cost, 1 << i, i, d, [(i, d)]))

    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_size]

    for _depth in range(1, m):
        next_beam = []
        for cost, mask, last_i, last_d, ordered in beam:
            for j in range(m):
                if mask & (1 << j):
                    continue
                for d2 in (0, 1):
                    new_cost = (
                        cost
                        + _bridge_cost(strokes[last_i], last_d, strokes[j], d2, weights)
                        + _internal_cost(strokes[j], d2, weights)
                    )
                    next_beam.append((
                        new_cost, mask | (1 << j), j, d2, ordered + [(j, d2)],
                    ))
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_size]

    if not beam:
        return StrokeRoute([], 0.0, (0, 0), (0, 0))

    best = beam[0]
    cost, _, _, _, ordered = best
    fi, fd = ordered[0]
    li, ld = ordered[-1]
    return StrokeRoute(
        ordered=ordered,
        total_cost=cost,
        start_point=strokes[fi].start(fd),
        end_point=strokes[li].end(ld),
    )


def route_word_strokes(
    strokes: List[PrimitiveStroke],
    weights: RouteWeights,
    dp_threshold: int = 16,
    beam_size: int = 32,
) -> StrokeRoute:
    """Route word strokes: exact DP if small enough, beam search otherwise."""
    if len(strokes) <= dp_threshold:
        return route_word_strokes_dp(strokes, weights)
    return route_word_strokes_beam(strokes, weights, beam_size)


# ---------------------------------------------------------------------------
# Event stream generation
# ---------------------------------------------------------------------------

def route_to_events(
    route: StrokeRoute,
    strokes: List[PrimitiveStroke],
    resample_step: float = 0.005,
) -> List[Dict[str, Any]]:
    """Convert a stroke route to the unified pen event format.

    Events: {dx, dy, pen_down, stroke_end, char_end, word_end, seq_end}
    """
    if not route.ordered:
        return []

    events = []
    prev_x, prev_y = 0.0, 0.0
    prev_char_id = None

    for si, (stroke_idx, direction) in enumerate(route.ordered):
        stroke = strokes[stroke_idx]
        pts = stroke.points(direction)
        if resample_step > 0 and len(pts) > 2:
            pts = resample_polyline(pts, resample_step)

        is_last_stroke = si == len(route.ordered) - 1
        char_changed = prev_char_id is not None and stroke.char_id != prev_char_id
        # Look ahead to see if char changes after this stroke
        next_char_id = None
        if not is_last_stroke:
            next_stroke_idx, _ = route.ordered[si + 1]
            next_char_id = strokes[next_stroke_idx].char_id
        is_char_end = is_last_stroke or next_char_id != stroke.char_id

        for pi, (px, py) in enumerate(pts):
            is_first = pi == 0
            is_last = pi == len(pts) - 1

            dx = round(px - prev_x, 6)
            dy = round(py - prev_y, 6)

            if si == 0 and pi == 0:
                # First point
                events.append({
                    "dx": 0.0, "dy": 0.0,
                    "pen_down": 1, "stroke_end": 0,
                    "char_end": 0, "word_end": 0, "seq_end": 0,
                })
                prev_x, prev_y = px, py
                prev_char_id = stroke.char_id
                continue

            if is_first:
                # Pen-up travel to stroke start
                events.append({
                    "dx": dx, "dy": dy,
                    "pen_down": 0, "stroke_end": 0,
                    "char_end": int(char_changed), "word_end": 0, "seq_end": 0,
                })
            else:
                events.append({
                    "dx": dx, "dy": dy,
                    "pen_down": 1,
                    "stroke_end": int(is_last),
                    "char_end": int(is_last and is_char_end),
                    "word_end": int(is_last and is_last_stroke),
                    "seq_end": int(is_last and is_last_stroke),
                })

            prev_x, prev_y = px, py
            prev_char_id = stroke.char_id

    return events
