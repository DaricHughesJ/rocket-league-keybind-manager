#!/usr/bin/env python3
"""
Shape Detection & Predictive Multi-Step Vector Collision Engine
================================================================
Principal Computer Vision / Geometry Tracking Script

Captures a desktop region at ~60 FPS via mss, detects a rectangular
workspace boundary (Canny + HoughLines), tracks circles (HoughCircles),
and runs a fully predictive cue→target collision calculation core:

  1. Initial vector ray from cue A toward target B
  2. At radius-boundary intercept: split into
       Ray 1 — target forward-momentum path
       Ray 2 — cue tangent-deflection path
  3. Recursive elastic wall reflections (∠in = ∠out) on both secondary rays
  4. Combined multi-segment trajectory array for advance outcome rendering

Dependencies:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shape_tracker")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point = Tuple[float, float]
PathNodes = List[Point]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Circle:
    """Detected circle in image / screen space."""

    cx: float
    cy: float
    radius: float

    @property
    def center(self) -> Point:
        return (self.cx, self.cy)

    def contains_point(self, pt: Point, margin: float = 0.0) -> bool:
        dx = pt[0] - self.cx
        dy = pt[1] - self.cy
        return (dx * dx + dy * dy) <= (self.radius + margin) ** 2

    def distance_to(self, other: "Circle") -> float:
        return math.hypot(self.cx - other.cx, self.cy - other.cy)


@dataclass
class Rectangle:
    """Axis-aligned workspace boundary rectangle (x_min, y_min, x_max, y_max)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def corners(self) -> List[Point]:
        return [
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
        ]

    def contains(self, pt: Point, margin: float = 0.0) -> bool:
        return (
            self.x_min + margin <= pt[0] <= self.x_max - margin
            and self.y_min + margin <= pt[1] <= self.y_max - margin
        )

    def clamp(self, pt: Point) -> Point:
        return (
            max(self.x_min, min(self.x_max, pt[0])),
            max(self.y_min, min(self.y_max, pt[1])),
        )


@dataclass
class CaptureConfig:
    """Desktop region to capture (mss monitor dict style)."""

    left: int = 0
    top: int = 0
    width: int = 1280
    height: int = 720
    target_fps: float = 60.0

    def to_mss_region(self) -> dict:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass
class DetectionConfig:
    """Tunable parameters for edge / line / circle detection."""

    # Canny
    canny_low: int = 50
    canny_high: int = 150
    blur_ksize: int = 5

    # HoughLinesP (for rectangle reconstruction)
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180.0
    hough_threshold: int = 80
    hough_min_line_length: int = 80
    hough_max_line_gap: int = 20

    # HoughCircles
    circle_dp: float = 1.2
    circle_min_dist: float = 30.0
    circle_param1: float = 100.0
    circle_param2: float = 30.0
    circle_min_radius: int = 8
    circle_max_radius: int = 120

    # Geometry
    line_angle_tolerance_deg: float = 8.0
    max_wall_bounces: int = 3
    ray_max_length: float = 5000.0
    epsilon: float = 1e-9


@dataclass
class RayPath:
    """One labeled polyline in the predictive collision sequence."""

    name: str
    nodes: PathNodes = field(default_factory=list)
    # BGR draw hint for the visual rendering engine
    color_bgr: Tuple[int, int, int] = (255, 255, 255)

    def as_array(self) -> List[Tuple[float, float]]:
        return list(self.nodes)


@dataclass
class CollisionPrediction:
    """
    Full predictive physics outcome for cue A → target B.

    `combined_rays` holds every segment the renderer should draw simultaneously:
      - initial      : A → impact
      - target       : Ray 1 (B forward momentum, with wall bounces)
      - deflection   : Ray 2 (A tangent deflection, with wall bounces)
    """

    cue: Circle
    target: Circle
    impact_point: Optional[Point] = None
    impact_normal: Optional[Point] = None
    initial_ray: PathNodes = field(default_factory=list)
    target_ray: PathNodes = field(default_factory=list)       # Ray 1
    deflection_ray: PathNodes = field(default_factory=list)  # Ray 2
    combined_rays: List[RayPath] = field(default_factory=list)
    # Flattened node list across all rays (ray breaks marked by None separators
    # are omitted here; use combined_rays for structured draw). Also exposed as
    # a single concatenated polyline list for simple consumers.
    combined_nodes: PathNodes = field(default_factory=list)
    did_collide: bool = False

    def to_render_array(self) -> List[Dict]:
        """
        End-result rendering payload: all rays combined for simultaneous draw.

        Returns
        -------
        [
          {"name": "initial",    "nodes": [(x,y), ...]},
          {"name": "target",     "nodes": [(x,y), ...]},
          {"name": "deflection", "nodes": [(x,y), ...]},
        ]
        """
        return [
            {"name": ray.name, "nodes": list(ray.nodes), "color_bgr": list(ray.color_bgr)}
            for ray in self.combined_rays
            if len(ray.nodes) >= 1
        ]


@dataclass
class FrameResult:
    """Per-frame detection + predictive collision output."""

    frame_bgr: np.ndarray
    boundary: Optional[Rectangle]
    circles: List[Circle] = field(default_factory=list)
    ray_path: PathNodes = field(default_factory=list)
    prediction: Optional[CollisionPrediction] = None
    fps: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Vector mathematics helpers
# ---------------------------------------------------------------------------
def _v_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _v_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = _v_norm(v)
    if n < eps:
        return np.zeros(2, dtype=np.float64)
    return v / n


def _v_reflect(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Reflect incident direction off a surface with unit normal.
    Angle In = Angle Out: R = I - 2 (I · N) N
    """
    n = _v_normalize(normal)
    i = _v_normalize(incident)
    return _v_normalize(i - 2.0 * float(np.dot(i, n)) * n)


def _v_perp(v: np.ndarray) -> np.ndarray:
    """Rotate 2D vector 90 degrees CCW (leftward tangent)."""
    return np.array([-v[1], v[0]], dtype=np.float64)


def _point_to_ndarray(pt: Point) -> np.ndarray:
    return np.array([pt[0], pt[1]], dtype=np.float64)


def _ndarray_to_point(v: np.ndarray) -> Point:
    return (float(v[0]), float(v[1]))


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------
class ScreenCapture:
    """Continuous mss-based screen grabber targeting a fixed FPS budget."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._sct: Optional[mss.mss] = None
        self._frame_interval = 1.0 / max(config.target_fps, 1.0)
        self._last_grab_time = 0.0

    def __enter__(self) -> "ScreenCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._sct is None:
            try:
                self._sct = mss.mss()
                logger.info(
                    "Screen capture opened: region=%s @ %.1f FPS",
                    self.config.to_mss_region(),
                    self.config.target_fps,
                )
            except Exception as exc:
                logger.error("Failed to open mss screen capture: %s", exc)
                raise RuntimeError(f"mss initialization failed: {exc}") from exc

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception as exc:
                logger.warning("Error closing mss: %s", exc)
            finally:
                self._sct = None

    def grab(self) -> np.ndarray:
        """
        Grab one BGR frame from the configured desktop region.
        Pace the call to approximate target_fps.
        """
        if self._sct is None:
            raise RuntimeError("ScreenCapture is not open. Call open() or use as context manager.")

        # Pace to target FPS
        now = time.perf_counter()
        elapsed = now - self._last_grab_time
        sleep_for = self._frame_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        try:
            region = self.config.to_mss_region()
            if region["width"] <= 0 or region["height"] <= 0:
                raise ValueError(f"Invalid capture region dimensions: {region}")

            shot = self._sct.grab(region)
            # mss returns BGRA
            frame = np.asarray(shot, dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise RuntimeError(f"Unexpected frame shape from mss: {frame.shape}")

            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self._last_grab_time = time.perf_counter()
            return bgr
        except Exception as exc:
            logger.error("Screen grab failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Geometry detection
# ---------------------------------------------------------------------------
class GeometryDetector:
    """Detect rectangular workspace boundary and circles inside it."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()

    # ---- preprocessing ----------------------------------------------------
    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame passed to GeometryDetector")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        k = self.config.blur_ksize
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)
        return gray, edges

    # ---- rectangle from Hough lines ---------------------------------------
    def detect_boundary_rectangle(
        self, frame_bgr: np.ndarray
    ) -> Optional[Rectangle]:
        """
        Map a perfect rectangular bounding box from Canny edges + HoughLinesP.
        Strategy: cluster near-horizontal / near-vertical segments, then take
        the outermost extents as the workspace rectangle.
        """
        try:
            _, edges = self._preprocess(frame_bgr)
            lines = cv2.HoughLinesP(
                edges,
                rho=self.config.hough_rho,
                theta=self.config.hough_theta,
                threshold=self.config.hough_threshold,
                minLineLength=self.config.hough_min_line_length,
                maxLineGap=self.config.hough_max_line_gap,
            )

            if lines is None or len(lines) == 0:
                # Fallback: use full frame as boundary
                h, w = frame_bgr.shape[:2]
                logger.debug("No Hough lines; using full-frame boundary %dx%d", w, h)
                return Rectangle(0.0, 0.0, float(w - 1), float(h - 1))

            # OpenCV may return (N,1,4) or (N,4) depending on version/backend
            segments = np.asarray(lines, dtype=np.float64).reshape(-1, 4)

            tol = math.radians(self.config.line_angle_tolerance_deg)
            horiz_ys: List[float] = []
            vert_xs: List[float] = []

            for x1, y1, x2, y2 in segments:
                dx, dy = float(x2 - x1), float(y2 - y1)
                angle = abs(math.atan2(dy, dx))
                # Normalize to [0, pi/2]
                if angle > math.pi / 2:
                    angle = math.pi - angle

                length = math.hypot(dx, dy)
                if length < 1.0:
                    continue

                if angle < tol:
                    # Horizontal
                    horiz_ys.append((float(y1) + float(y2)) * 0.5)
                elif abs(angle - math.pi / 2) < tol:
                    # Vertical
                    vert_xs.append((float(x1) + float(x2)) * 0.5)

            h, w = frame_bgr.shape[:2]

            if len(horiz_ys) < 2 or len(vert_xs) < 2:
                # Not enough axis-aligned edges — fall back to AABB of endpoints
                x_min = float(np.min(segments[:, [0, 2]]))
                x_max = float(np.max(segments[:, [0, 2]]))
                y_min = float(np.min(segments[:, [1, 3]]))
                y_max = float(np.max(segments[:, [1, 3]]))
            else:
                # Outermost horizontal and vertical lines form the box
                y_min = float(min(horiz_ys))
                y_max = float(max(horiz_ys))
                x_min = float(min(vert_xs))
                x_max = float(max(vert_xs))

            # Sanity clamp to image
            x_min = max(0.0, min(x_min, w - 1.0))
            x_max = max(0.0, min(x_max, w - 1.0))
            y_min = max(0.0, min(y_min, h - 1.0))
            y_max = max(0.0, min(y_max, h - 1.0))

            if x_max - x_min < 10 or y_max - y_min < 10:
                logger.warning("Degenerate rectangle detected; using full frame")
                return Rectangle(0.0, 0.0, float(w - 1), float(h - 1))

            return Rectangle(x_min, y_min, x_max, y_max)

        except Exception as exc:
            logger.error("Boundary rectangle detection failed: %s", exc)
            return None

    # ---- circles ----------------------------------------------------------
    def detect_circles(
        self,
        frame_bgr: np.ndarray,
        boundary: Optional[Rectangle] = None,
    ) -> List[Circle]:
        """Detect circle centers and radii via HoughCircles, filtered by boundary."""
        try:
            gray, _ = self._preprocess(frame_bgr)
            circles_raw = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=self.config.circle_dp,
                minDist=self.config.circle_min_dist,
                param1=self.config.circle_param1,
                param2=self.config.circle_param2,
                minRadius=self.config.circle_min_radius,
                maxRadius=self.config.circle_max_radius,
            )

            if circles_raw is None:
                return []

            results: List[Circle] = []
            for c in circles_raw[0]:
                cx, cy, r = float(c[0]), float(c[1]), float(c[2])
                circ = Circle(cx, cy, r)
                if boundary is not None:
                    # Keep only circles whose center lies inside the workspace
                    if not boundary.contains(circ.center):
                        continue
                results.append(circ)

            # Sort left-to-right, top-to-bottom for stable indexing
            results.sort(key=lambda c: (c.cy, c.cx))
            return results

        except Exception as exc:
            logger.error("Circle detection failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Recursive ray-tracing physics engine
# ---------------------------------------------------------------------------
class RayTracingEngine:
    """
    Vector mathematics engine for multi-bounce path prediction.

    - Wall hit  → recursive reflection (Angle In = Angle Out), up to N bounces
    - Circle hit → compute split tangent vectors at contact; continue along
                   the chosen tangent branch (primary = closer to original dir)
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()

    def compute_path(
        self,
        origin: Point,
        direction: Sequence[float],
        boundary: Rectangle,
        circles: Sequence[Circle],
        moving_circle_index: Optional[int] = None,
        max_bounces: Optional[int] = None,
    ) -> PathNodes:
        """
        Trace a ray from `origin` along `direction` and return polyline nodes.

        Parameters
        ----------
        origin : starting point (typically a circle center)
        direction : 2D direction vector (need not be unit length)
        boundary : workspace rectangle walls
        circles : all detected circles (obstacles / peers)
        moving_circle_index : if set, exclude this circle from collision tests
                              (the object emitting the ray)
        max_bounces : override for consecutive wall bounce limit (default 3)

        Returns
        -------
        List of (x, y) nodes forming the multi-bounce path.
        """
        try:
            if boundary is None:
                raise ValueError("boundary rectangle is required")

            max_b = (
                self.config.max_wall_bounces
                if max_bounces is None
                else int(max_bounces)
            )
            max_b = max(0, min(max_b, 3))  # hard-cap at 3 consecutive wall bounces

            dir_v = _v_normalize(np.asarray(direction, dtype=np.float64))
            if _v_norm(dir_v) < self.config.epsilon:
                logger.warning("Zero-length direction; returning origin only")
                return [origin]

            obstacles = [
                c
                for i, c in enumerate(circles)
                if moving_circle_index is None or i != moving_circle_index
            ]

            nodes: PathNodes = [origin]
            pos = _point_to_ndarray(origin)
            direction_v = dir_v.copy()
            remaining_length = float(self.config.ray_max_length)
            wall_bounces = 0

            self._trace_recursive(
                pos=pos,
                direction=direction_v,
                boundary=boundary,
                obstacles=obstacles,
                nodes=nodes,
                wall_bounces=wall_bounces,
                max_wall_bounces=max_b,
                remaining_length=remaining_length,
                consumed_circles=set(),
            )
            return nodes

        except Exception as exc:
            logger.error("Ray path computation failed: %s", exc)
            return [origin]

    @staticmethod
    def _circle_key(c: Circle) -> Tuple[float, float, float]:
        return (round(c.cx, 3), round(c.cy, 3), round(c.radius, 3))

    # ---- recursive core ---------------------------------------------------
    def _trace_recursive(
        self,
        pos: np.ndarray,
        direction: np.ndarray,
        boundary: Rectangle,
        obstacles: Sequence[Circle],
        nodes: PathNodes,
        wall_bounces: int,
        max_wall_bounces: int,
        remaining_length: float,
        consumed_circles: Optional[set] = None,
        depth: int = 0,
    ) -> None:
        """Recursively advance the ray until length/bounce budget exhausted."""
        MAX_DEPTH = 16  # safety against infinite recursion
        if depth > MAX_DEPTH or remaining_length <= self.config.epsilon:
            return

        direction = _v_normalize(direction)
        if _v_norm(direction) < self.config.epsilon:
            return

        consumed = consumed_circles if consumed_circles is not None else set()
        active_obstacles = [
            c for c in obstacles if self._circle_key(c) not in consumed
        ]

        # Find nearest intersection: wall or circle
        wall_hit = self._intersect_walls(pos, direction, boundary)
        circle_hit = self._intersect_circles(pos, direction, active_obstacles)

        # Choose nearer positive hit
        candidates: List[Tuple[float, str, object]] = []
        if wall_hit is not None:
            candidates.append((wall_hit[0], "wall", wall_hit))
        if circle_hit is not None:
            candidates.append((circle_hit[0], "circle", circle_hit))

        if not candidates:
            # No hit — extend to remaining length and stop (clamped inside bounds)
            end = pos + direction * remaining_length
            end = _point_to_ndarray(boundary.clamp(_ndarray_to_point(end)))
            nodes.append(_ndarray_to_point(end))
            return

        candidates.sort(key=lambda t: t[0])
        dist, kind, payload = candidates[0]

        if dist > remaining_length:
            end = pos + direction * remaining_length
            end = _point_to_ndarray(boundary.clamp(_ndarray_to_point(end)))
            nodes.append(_ndarray_to_point(end))
            return

        if kind == "wall":
            _, hit_pt, normal = payload  # type: ignore[misc]
            hit_pt = np.asarray(hit_pt, dtype=np.float64)
            # Project onto the wall plane, then nudge inward (opposite outward normal)
            inward = -_v_normalize(np.asarray(normal, dtype=np.float64))
            hit_pt = _point_to_ndarray(boundary.clamp(_ndarray_to_point(hit_pt)))
            hit_pt = hit_pt + inward * 1e-3
            hit_pt = _point_to_ndarray(boundary.clamp(_ndarray_to_point(hit_pt)))
            nodes.append(_ndarray_to_point(hit_pt))

            if wall_bounces >= max_wall_bounces:
                return

            reflected = _v_reflect(direction, np.asarray(normal, dtype=np.float64))
            self._trace_recursive(
                pos=hit_pt,
                direction=reflected,
                boundary=boundary,
                obstacles=obstacles,
                nodes=nodes,
                wall_bounces=wall_bounces + 1,
                max_wall_bounces=max_wall_bounces,
                remaining_length=remaining_length - dist,
                consumed_circles=consumed,
                depth=depth + 1,
            )
            return

        # Circle collision → split tangent vectors
        _, hit_pt, circle = payload  # type: ignore[misc]
        hit_pt = np.asarray(hit_pt, dtype=np.float64)
        nodes.append(_ndarray_to_point(hit_pt))

        left_t, right_t = self._split_tangent_vectors(hit_pt, circle)
        # Continue along the tangent closer to the original direction
        if float(np.dot(left_t, direction)) >= float(np.dot(right_t, direction)):
            next_dir = left_t
        else:
            next_dir = right_t

        next_consumed = set(consumed)
        next_consumed.add(self._circle_key(circle))

        # Advance past contact; do not re-collide with this circle
        advance = hit_pt + next_dir * max(1e-3, circle.radius * 0.02)
        self._trace_recursive(
            pos=advance,
            direction=next_dir,
            boundary=boundary,
            obstacles=obstacles,
            nodes=nodes,
            # Consecutive wall-bounce counter resets after a circle interaction
            wall_bounces=0,
            max_wall_bounces=max_wall_bounces,
            remaining_length=remaining_length - dist,
            consumed_circles=next_consumed,
            depth=depth + 1,
        )

    # ---- wall intersection ------------------------------------------------
    def _intersect_walls(
        self,
        pos: np.ndarray,
        direction: np.ndarray,
        boundary: Rectangle,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Ray–AABB side intersection.
        Returns (distance, hit_point, outward_normal) or None.
        """
        eps = self.config.epsilon
        best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

        # Four walls: left, right, top, bottom with outward normals
        walls = [
            # left x = x_min, normal (-1, 0)
            (boundary.x_min, True, np.array([-1.0, 0.0])),
            # right x = x_max, normal (1, 0)
            (boundary.x_max, True, np.array([1.0, 0.0])),
            # top y = y_min, normal (0, -1)
            (boundary.y_min, False, np.array([0.0, -1.0])),
            # bottom y = y_max, normal (0, 1)
            (boundary.y_max, False, np.array([0.0, 1.0])),
        ]

        for plane_val, is_vertical, normal in walls:
            if is_vertical:
                denom = direction[0]
                if abs(denom) < eps:
                    continue
                t = (plane_val - pos[0]) / denom
            else:
                denom = direction[1]
                if abs(denom) < eps:
                    continue
                t = (plane_val - pos[1]) / denom

            if t <= eps:
                continue

            hit = pos + direction * t
            # Must lie on the finite wall segment (with small tolerance)
            if is_vertical:
                if hit[1] < boundary.y_min - 1.0 or hit[1] > boundary.y_max + 1.0:
                    continue
            else:
                if hit[0] < boundary.x_min - 1.0 or hit[0] > boundary.x_max + 1.0:
                    continue

            # Only count hits where we are traveling into the wall
            if float(np.dot(direction, normal)) <= 0:
                # Traveling toward interior or parallel — skip
                # For reflection we need outward-facing hits from inside
                # Ray from inside: direction · outward_normal > 0
                pass
            if float(np.dot(direction, normal)) <= eps:
                continue

            if best is None or t < best[0]:
                best = (float(t), hit.copy(), normal.astype(np.float64))

        return best

    # ---- circle intersection ----------------------------------------------
    def _intersect_circles(
        self,
        pos: np.ndarray,
        direction: np.ndarray,
        obstacles: Sequence[Circle],
    ) -> Optional[Tuple[float, np.ndarray, Circle]]:
        """
        Nearest forward ray–circle intersection.
        Returns (distance, hit_point, circle) or None.
        """
        eps = self.config.epsilon
        best: Optional[Tuple[float, np.ndarray, Circle]] = None
        d = direction  # unit

        for circ in obstacles:
            center = _point_to_ndarray(circ.center)
            oc = pos - center
            # Quadratic: |oc + t d|^2 = r^2
            a = float(np.dot(d, d))
            b = 2.0 * float(np.dot(oc, d))
            c = float(np.dot(oc, oc)) - circ.radius ** 2
            disc = b * b - 4.0 * a * c
            if disc < 0:
                continue
            sqrt_disc = math.sqrt(disc)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)
            t = None
            if t0 > eps:
                t = t0
            elif t1 > eps:
                t = t1
            if t is None:
                continue
            hit = pos + d * t
            if best is None or t < best[0]:
                best = (float(t), hit.copy(), circ)

        return best

    # ---- split tangents at circle contact ---------------------------------
    @staticmethod
    def _split_tangent_vectors(
        hit_pt: np.ndarray, circle: Circle
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        At contact point, compute the two unit tangent vectors
        (perpendicular to the radius / surface normal).
        """
        radial = hit_pt - _point_to_ndarray(circle.center)
        normal = _v_normalize(radial)
        if _v_norm(normal) < 1e-12:
            # Degenerate — arbitrary basis
            return (
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([-1.0, 0.0], dtype=np.float64),
            )
        left = _v_normalize(_v_perp(normal))
        right = -left
        return left, right

    def trace_wall_reflections(
        self,
        origin: Point,
        direction: Sequence[float],
        boundary: Rectangle,
        max_bounces: Optional[int] = None,
    ) -> PathNodes:
        """
        Trace a ray with recursive elastic wall reflections only
        (Angle of Incidence = Angle of Reflection). No circle interactions.
        """
        return self.compute_path(
            origin=origin,
            direction=direction,
            boundary=boundary,
            circles=[],
            moving_circle_index=None,
            max_bounces=max_bounces,
        )


# ---------------------------------------------------------------------------
# Predictive multi-step vector collision engine (calculation core)
# ---------------------------------------------------------------------------
class PredictiveCollisionEngine:
    """
    Fully predictive cue→target collision calculator on a 2D canvas.

    Sequence
    --------
    1. Initial Vector — continuous ray from cue A along an aim angle toward B
    2. Interception   — stop at the exact intersection with B's collision radius
    3. Split:
         Ray 1 (Target Path)      — B's forward-momentum away from impact
         Ray 2 (Deflection Path)  — A's tangent deflection after collision
    4. Boundary Reflection — recursive ∠in=∠out wall bounces on both rays
    5. Combined render array — all segments for simultaneous trajectory draw
    """

    # Draw colors (BGR)
    COLOR_INITIAL = (80, 80, 255)      # red-ish: cue approach
    COLOR_TARGET = (80, 220, 80)       # green: target momentum
    COLOR_DEFLECTION = (255, 180, 60)  # cyan-ish: cue deflection

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()
        self.ray_engine = RayTracingEngine(self.config)

    def predict(
        self,
        cue: Circle,
        target: Circle,
        boundary: Rectangle,
        aim_direction: Optional[Sequence[float]] = None,
        max_bounces: Optional[int] = None,
    ) -> CollisionPrediction:
        """
        Compute the full multi-step predictive collision outcome.

        Parameters
        ----------
        cue : Circle A (cue / striking object)
        target : Circle B (struck object)
        boundary : outer rectangular walls
        aim_direction : optional aim vector; defaults to (B - A)
        max_bounces : wall bounce budget for each secondary ray (default ≤ 3)
        """
        try:
            return self._predict_impl(
                cue, target, boundary, aim_direction, max_bounces
            )
        except Exception as exc:
            logger.error("Predictive collision failed: %s", exc)
            return CollisionPrediction(
                cue=cue,
                target=target,
                initial_ray=[cue.center],
                combined_rays=[
                    RayPath("initial", [cue.center], self.COLOR_INITIAL)
                ],
                combined_nodes=[cue.center],
                did_collide=False,
            )

    def _predict_impl(
        self,
        cue: Circle,
        target: Circle,
        boundary: Rectangle,
        aim_direction: Optional[Sequence[float]],
        max_bounces: Optional[int],
    ) -> CollisionPrediction:
        eps = self.config.epsilon
        max_b = (
            self.config.max_wall_bounces
            if max_bounces is None
            else int(max_bounces)
        )
        max_b = max(0, min(max_b, 3))

        a = _point_to_ndarray(cue.center)
        b = _point_to_ndarray(target.center)

        # ---- 1. Initial vector (aim from A toward B) ----------------------
        if aim_direction is not None:
            aim = _v_normalize(np.asarray(aim_direction, dtype=np.float64))
        else:
            aim = _v_normalize(b - a)

        if _v_norm(aim) < eps:
            # Degenerate: A and B coincide / zero aim
            pred = CollisionPrediction(
                cue=cue,
                target=target,
                initial_ray=[cue.center],
                did_collide=False,
            )
            pred.combined_rays = [
                RayPath("initial", [cue.center], self.COLOR_INITIAL)
            ]
            pred.combined_nodes = [cue.center]
            return pred

        # Collision radius: centers contact when |A-B| = rA + rB.
        # Ray of cue center therefore intersects a circle about B of that radius.
        collision_radius = float(cue.radius + target.radius)
        if collision_radius <= eps:
            collision_radius = float(max(target.radius, 1.0))

        hit = self._intersect_ray_circle(a, aim, b, collision_radius)

        if hit is None:
            # No interception — extend initial ray with wall reflections only
            initial = self.ray_engine.trace_wall_reflections(
                origin=cue.center,
                direction=aim,
                boundary=boundary,
                max_bounces=max_b,
            )
            pred = CollisionPrediction(
                cue=cue,
                target=target,
                initial_ray=initial,
                did_collide=False,
            )
            pred.combined_rays = [
                RayPath("initial", initial, self.COLOR_INITIAL)
            ]
            pred.combined_nodes = list(initial)
            return pred

        t_hit, impact_pos = hit
        impact_point = _ndarray_to_point(impact_pos)

        # Cue center at contact; surface contact on B lies along impact normal
        # Impact normal: from B through contact geometry (A→B at impact)
        # At contact, cue center is at impact_pos; B is still at b
        # Line of centres: n = normalize(b_contact_direction)
        # Using geometry: n points from A toward B along centres = normalize(b - impact_pos)
        # Actually at contact: impact_pos is cue center; target center is b;
        # n_impact (A→B) = normalize(b - impact_pos)
        n_ab = _v_normalize(b - impact_pos)
        if _v_norm(n_ab) < eps:
            n_ab = aim.copy()

        # Surface intersection on B's radius boundary (visual impact marker)
        surface_hit = b - n_ab * float(target.radius)
        surface_point = _ndarray_to_point(surface_hit)

        # Prefer the radius-boundary point of B as the documented split origin
        split_origin = surface_point

        # Initial ray: A center → stop at B radius-boundary intercept
        initial_ray: PathNodes = [cue.center, split_origin]

        # ---- 2/3. Interception & deflection split -------------------------
        # Equal-mass elastic collision, B initially at rest:
        #   v_n = (aim · n) n     → transferred to B  (Ray 1: target path)
        #   v_t = aim - v_n       → retained by A     (Ray 2: deflection)
        # where n is the impact-axis unit vector (A→B / away from impact into B).
        n = n_ab  # forward direction for target momentum (away from cue)
        vn_scale = float(np.dot(aim, n))
        # Only compressive (approaching) normal component transfers
        if vn_scale < 0:
            # Aiming away from target along normal — flip
            n = -n
            vn_scale = float(np.dot(aim, n))

        v_normal = n * max(vn_scale, 0.0)
        v_tangent = aim - v_normal

        # Ray 1 — Target Path: forward momentum of B away from impact
        target_dir = _v_normalize(v_normal)
        if _v_norm(target_dir) < eps:
            # Glancing / pure-tangent hit: B receives negligible normal impulse
            target_dir = n.copy()

        # Ray 2 — Deflection Path: tangent deflection of A after collision
        deflect_dir = _v_normalize(v_tangent)
        if _v_norm(deflect_dir) < eps:
            # Head-on: A stops (no tangent component) — leave empty / zero ray
            deflect_dir = np.zeros(2, dtype=np.float64)

        # Also expose classical left/right geometric tangents at the contact
        # for consumers that need the full split-tangent pair.
        _left_t, _right_t = self.ray_engine._split_tangent_vectors(
            surface_hit, target
        )

        # ---- 4. Boundary reflection on both secondary rays ----------------
        # Start secondary rays from the split origin, nudged along each dir
        # so the first sample is not stuck on the surface.
        nudge = max(1.0, target.radius * 0.05)

        if _v_norm(target_dir) >= eps:
            t_start = _ndarray_to_point(
                _point_to_ndarray(split_origin) + target_dir * nudge
            )
            t_start = boundary.clamp(t_start)
            target_ray = self.ray_engine.trace_wall_reflections(
                origin=t_start,
                direction=target_dir,
                boundary=boundary,
                max_bounces=max_b,
            )
            # Prepend exact impact/split point for continuous polyline
            if not target_ray or target_ray[0] != split_origin:
                target_ray = [split_origin] + list(target_ray)
        else:
            target_ray = [split_origin]

        if _v_norm(deflect_dir) >= eps:
            d_start = _ndarray_to_point(
                _point_to_ndarray(split_origin) + deflect_dir * nudge
            )
            d_start = boundary.clamp(d_start)
            deflection_ray = self.ray_engine.trace_wall_reflections(
                origin=d_start,
                direction=deflect_dir,
                boundary=boundary,
                max_bounces=max_b,
            )
            if not deflection_ray or deflection_ray[0] != split_origin:
                deflection_ray = [split_origin] + list(deflection_ray)
        else:
            # Head-on: cue deposits all momentum — deflection ray collapses
            deflection_ray = [split_origin]

        # ---- 5. Combined render array -------------------------------------
        combined_rays = [
            RayPath("initial", initial_ray, self.COLOR_INITIAL),
            RayPath("target", target_ray, self.COLOR_TARGET),
            RayPath("deflection", deflection_ray, self.COLOR_DEFLECTION),
        ]
        combined_nodes: PathNodes = []
        for ray in combined_rays:
            combined_nodes.extend(ray.nodes)

        return CollisionPrediction(
            cue=cue,
            target=target,
            impact_point=split_origin,
            impact_normal=_ndarray_to_point(n),
            initial_ray=initial_ray,
            target_ray=target_ray,
            deflection_ray=deflection_ray,
            combined_rays=combined_rays,
            combined_nodes=combined_nodes,
            did_collide=True,
        )

    @staticmethod
    def _intersect_ray_circle(
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
        eps: float = 1e-9,
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Nearest forward intersection of ray (origin + t*direction) with circle.
        Returns (t, point) or None.
        """
        d = _v_normalize(direction)
        if _v_norm(d) < eps:
            return None
        oc = origin - center
        a = float(np.dot(d, d))
        b = 2.0 * float(np.dot(oc, d))
        c = float(np.dot(oc, oc)) - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t = None
        if t0 > eps:
            t = t0
        elif t1 > eps:
            t = t1
        if t is None:
            return None
        return float(t), origin + d * t

    def predict_from_indices(
        self,
        circles: Sequence[Circle],
        boundary: Rectangle,
        cue_index: int = 0,
        target_index: int = 1,
        aim_direction: Optional[Sequence[float]] = None,
        max_bounces: Optional[int] = None,
    ) -> CollisionPrediction:
        """Convenience wrapper selecting cue/target by list index."""
        if len(circles) < 2:
            raise ValueError("Need at least two circles (cue and target)")
        if cue_index < 0 or cue_index >= len(circles):
            raise IndexError(f"cue_index {cue_index} out of range")
        if target_index < 0 or target_index >= len(circles):
            raise IndexError(f"target_index {target_index} out of range")
        if cue_index == target_index:
            raise ValueError("cue_index and target_index must differ")
        return self.predict(
            cue=circles[cue_index],
            target=circles[target_index],
            boundary=boundary,
            aim_direction=aim_direction,
            max_bounces=max_bounces,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class ShapeTracker:
    """
    High-level loop: capture → detect rectangle → detect circles →
    predictive cue/target collision ray bundle.
    """

    def __init__(
        self,
        capture_config: CaptureConfig,
        detection_config: Optional[DetectionConfig] = None,
        visualize: bool = True,
        ray_direction: Optional[Sequence[float]] = None,
        tracked_circle_index: int = 0,
        target_circle_index: int = 1,
    ) -> None:
        self.capture_config = capture_config
        self.detection_config = detection_config or DetectionConfig()
        self.visualize = visualize
        self.ray_direction = (
            list(ray_direction) if ray_direction is not None else None
        )
        self.tracked_circle_index = tracked_circle_index
        self.target_circle_index = target_circle_index

        self.detector = GeometryDetector(self.detection_config)
        self.ray_engine = RayTracingEngine(self.detection_config)
        self.collision_engine = PredictiveCollisionEngine(self.detection_config)
        self._running = False
        self._prev_centers: List[Point] = []

    def process_frame(self, frame_bgr: np.ndarray, fps: float = 0.0) -> FrameResult:
        """Run full detection + predictive collision pipeline on one BGR frame."""
        ts = time.time()
        boundary = self.detector.detect_boundary_rectangle(frame_bgr)
        circles = self.detector.detect_circles(frame_bgr, boundary)

        ray_path: PathNodes = []
        prediction: Optional[CollisionPrediction] = None

        if boundary is not None and len(circles) >= 2:
            cue_idx = self.tracked_circle_index
            tgt_idx = self.target_circle_index
            if cue_idx < 0 or cue_idx >= len(circles):
                cue_idx = 0
            if tgt_idx < 0 or tgt_idx >= len(circles) or tgt_idx == cue_idx:
                tgt_idx = 1 if cue_idx != 1 else 0

            # Infer aim from cue motion when available; else aim at target
            aim = self.ray_direction
            if self._prev_centers and cue_idx < len(self._prev_centers):
                prev = self._prev_centers[cue_idx]
                curr = circles[cue_idx].center
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                if math.hypot(dx, dy) > 1.0:
                    aim = [dx, dy]

            if aim is None:
                aim = [
                    circles[tgt_idx].cx - circles[cue_idx].cx,
                    circles[tgt_idx].cy - circles[cue_idx].cy,
                ]

            prediction = self.collision_engine.predict(
                cue=circles[cue_idx],
                target=circles[tgt_idx],
                boundary=boundary,
                aim_direction=aim,
            )
            ray_path = list(prediction.combined_nodes)

        elif boundary is not None and len(circles) == 1:
            # Single object: wall-bounce path only
            idx = 0
            direction = self.ray_direction or [1.0, 0.0]
            if self._prev_centers:
                prev = self._prev_centers[0]
                curr = circles[0].center
                dx, dy = curr[0] - prev[0], curr[1] - prev[1]
                if math.hypot(dx, dy) > 1.0:
                    direction = [dx, dy]
            ray_path = self.ray_engine.trace_wall_reflections(
                origin=circles[idx].center,
                direction=direction,
                boundary=boundary,
            )

        self._prev_centers = [c.center for c in circles]

        return FrameResult(
            frame_bgr=frame_bgr,
            boundary=boundary,
            circles=circles,
            ray_path=ray_path,
            prediction=prediction,
            fps=fps,
            timestamp=ts,
        )

    def run(self, max_frames: Optional[int] = None) -> PathNodes:
        """
        Continuous capture loop at configured FPS.
        Returns the last combined multi-segment trajectory node array.
        """
        last_path: PathNodes = []
        last_prediction: Optional[CollisionPrediction] = None
        frame_count = 0
        fps_clock = time.perf_counter()
        fps_counter = 0
        current_fps = 0.0

        try:
            with ScreenCapture(self.capture_config) as cap:
                self._running = True
                logger.info("Entering capture loop (Ctrl+C or 'q' to stop)")

                while self._running:
                    if max_frames is not None and frame_count >= max_frames:
                        break

                    try:
                        frame = cap.grab()
                    except Exception as exc:
                        logger.error("Grab error (continuing): %s", exc)
                        time.sleep(0.01)
                        continue

                    result = self.process_frame(frame, fps=current_fps)
                    last_path = result.ray_path
                    last_prediction = result.prediction
                    frame_count += 1
                    fps_counter += 1

                    now = time.perf_counter()
                    if now - fps_clock >= 1.0:
                        current_fps = fps_counter / (now - fps_clock)
                        fps_counter = 0
                        fps_clock = now
                        collided = (
                            result.prediction.did_collide
                            if result.prediction
                            else False
                        )
                        logger.info(
                            "FPS=%.1f | circles=%d | collide=%s | path_nodes=%d",
                            current_fps,
                            len(result.circles),
                            collided,
                            len(result.ray_path),
                        )

                    if self.visualize:
                        vis = self.render(result)
                        cv2.imshow("Shape Tracker / Predictive Collision", vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            logger.info("User quit requested")
                            break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as exc:
            logger.exception("Fatal error in capture loop: %s", exc)
            raise
        finally:
            self._running = False
            if self.visualize:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        if last_prediction is not None:
            logger.info(
                "Loop finished after %d frames. Render array: %s",
                frame_count,
                last_prediction.to_render_array(),
            )
        else:
            logger.info(
                "Loop finished after %d frames. Final path (%d nodes): %s",
                frame_count,
                len(last_path),
                last_path,
            )
        return last_path

    # ---- visualization ----------------------------------------------------
    @staticmethod
    def render(result: FrameResult) -> np.ndarray:
        """Overlay boundary, circles, and all predictive ray segments."""
        vis = result.frame_bgr.copy()

        if result.boundary is not None:
            b = result.boundary
            cv2.rectangle(
                vis,
                (int(b.x_min), int(b.y_min)),
                (int(b.x_max), int(b.y_max)),
                (0, 220, 0),
                2,
            )

        for i, c in enumerate(result.circles):
            center = (int(round(c.cx)), int(round(c.cy)))
            cv2.circle(vis, center, int(round(c.radius)), (0, 140, 255), 2)
            cv2.circle(vis, center, 3, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"C{i} ({c.cx:.0f},{c.cy:.0f}) r={c.radius:.0f}",
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        rays_to_draw: List[RayPath] = []
        if result.prediction is not None and result.prediction.combined_rays:
            rays_to_draw = result.prediction.combined_rays
            if result.prediction.impact_point is not None:
                ip = result.prediction.impact_point
                cv2.circle(
                    vis,
                    (int(round(ip[0])), int(round(ip[1]))),
                    6,
                    (0, 255, 255),
                    -1,
                )
        elif len(result.ray_path) >= 2:
            rays_to_draw = [RayPath("path", result.ray_path, (255, 80, 80))]

        for ray in rays_to_draw:
            if len(ray.nodes) < 2:
                continue
            pts = np.array(
                [[int(round(x)), int(round(y))] for x, y in ray.nodes],
                dtype=np.int32,
            )
            cv2.polylines(vis, [pts], False, ray.color_bgr, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(vis, tuple(p), 3, ray.color_bgr, -1)

        collide_flag = (
            "Y" if result.prediction and result.prediction.did_collide else "N"
        )
        hud = (
            f"FPS: {result.fps:.1f}  circles: {len(result.circles)}  "
            f"collide: {collide_flag}  nodes: {len(result.ray_path)}"
        )
        cv2.putText(
            vis, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 3, cv2.LINE_AA
        )
        cv2.putText(
            vis, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
        )
        # Legend
        legend = "initial=red  target=green  deflection=cyan"
        cv2.putText(
            vis, legend, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )
        return vis


# ---------------------------------------------------------------------------
# Public API convenience
# ---------------------------------------------------------------------------
def compute_multi_bounce_path(
    origin: Point,
    direction: Sequence[float],
    boundary: Rectangle,
    circles: Sequence[Circle],
    max_bounces: int = 3,
    moving_circle_index: Optional[int] = None,
) -> PathNodes:
    """
    Standalone entry point: return a multi-bounce path as
    [(x1,y1), (x2,y2), (x3,y3), ...].
    """
    engine = RayTracingEngine(DetectionConfig(max_wall_bounces=max_bounces))
    return engine.compute_path(
        origin=origin,
        direction=direction,
        boundary=boundary,
        circles=list(circles),
        moving_circle_index=moving_circle_index,
        max_bounces=max_bounces,
    )


def compute_collision_prediction(
    cue: Circle,
    target: Circle,
    boundary: Rectangle,
    aim_direction: Optional[Sequence[float]] = None,
    max_bounces: int = 3,
) -> CollisionPrediction:
    """
    Fully predictive cue→target collision.

    Returns a CollisionPrediction whose `to_render_array()` yields all rays
    combined for the visual rendering engine:

        [
          {"name": "initial",    "nodes": [(x,y), ...]},
          {"name": "target",     "nodes": [(x,y), ...]},
          {"name": "deflection", "nodes": [(x,y), ...]},
        ]
    """
    engine = PredictiveCollisionEngine(
        DetectionConfig(max_wall_bounces=max_bounces)
    )
    return engine.predict(
        cue=cue,
        target=target,
        boundary=boundary,
        aim_direction=aim_direction,
        max_bounces=max_bounces,
    )


# ---------------------------------------------------------------------------
# Synthetic demo (no display / no desktop content required)
# ---------------------------------------------------------------------------
def _build_synthetic_frame(
    width: int = 800,
    height: int = 600,
    circles: Optional[List[Circle]] = None,
) -> Tuple[np.ndarray, Rectangle, List[Circle]]:
    """Create a synthetic scene with a clear rectangle and circles for tests."""
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    margin = 40
    boundary = Rectangle(float(margin), float(margin), float(width - margin), float(height - margin))
    cv2.rectangle(
        frame,
        (int(boundary.x_min), int(boundary.y_min)),
        (int(boundary.x_max), int(boundary.y_max)),
        (220, 220, 220),
        3,
    )

    if circles is None:
        circles = [
            Circle(200.0, 200.0, 35.0),
            Circle(420.0, 280.0, 45.0),
            Circle(600.0, 180.0, 30.0),
        ]

    for c in circles:
        cv2.circle(frame, (int(c.cx), int(c.cy)), int(c.radius), (90, 180, 255), -1)
        cv2.circle(frame, (int(c.cx), int(c.cy)), int(c.radius), (255, 255, 255), 2)

    return frame, boundary, circles


def run_self_test() -> PathNodes:
    """
    Offline validation of detection + predictive collision engine.
    Returns the combined trajectory node array from the primary scenario.
    """
    logger.info("Running synthetic self-test…")
    frame, true_boundary, true_circles = _build_synthetic_frame()

    detector = GeometryDetector()
    detected_boundary = detector.detect_boundary_rectangle(frame)
    detected_circles = detector.detect_circles(frame, detected_boundary)

    logger.info("True boundary:      %s", true_boundary)
    logger.info("Detected boundary:  %s", detected_boundary)
    logger.info("True circles:       %s", true_circles)
    logger.info("Detected circles:   %s", detected_circles)

    boundary = detected_boundary or true_boundary
    circles = detected_circles if detected_circles else true_circles

    path = compute_multi_bounce_path(
        origin=circles[0].center,
        direction=(1.0, 0.35),
        boundary=boundary,
        circles=circles,
        max_bounces=3,
        moving_circle_index=0,
    )

    logger.info("Multi-bounce path (%d nodes): %s", len(path), path)

    # Basic invariants
    assert len(path) >= 2, "Path must contain at least origin and one hit/end"
    for node in path:
        assert isinstance(node, tuple) and len(node) == 2
        assert math.isfinite(node[0]) and math.isfinite(node[1])

    # Wall reflection unit test
    engine = RayTracingEngine()
    rect = Rectangle(0, 0, 100, 100)
    wall_path = engine.compute_path(
        origin=(50.0, 20.0),
        direction=(1.0, 1.0),
        boundary=rect,
        circles=[],
        max_bounces=3,
    )
    logger.info("Wall-only reflection path: %s", wall_path)
    assert len(wall_path) >= 2
    assert len(wall_path) <= 5, f"Too many wall nodes for 3-bounce cap: {wall_path}"
    for x, y in wall_path:
        assert -0.5 <= x <= 100.5 and -0.5 <= y <= 100.5

    # Circle split-tangent unit test (legacy single-path engine)
    circ = Circle(50.0, 50.0, 20.0)
    circle_path = engine.compute_path(
        origin=(10.0, 50.0),
        direction=(1.0, 0.0),
        boundary=rect,
        circles=[circ],
        max_bounces=3,
    )
    logger.info("Circle-collision path: %s", circle_path)
    assert len(circle_path) >= 2
    assert abs(circle_path[1][0] - 30.0) < 1.0
    assert len(circle_path) <= 6, f"Path oscillated too long: {circle_path}"

    path_multi = engine.compute_path(
        origin=(15.0, 20.0),
        direction=(1.0, 0.4),
        boundary=rect,
        circles=[Circle(40.0, 35.0, 12.0), Circle(70.0, 60.0, 10.0)],
        moving_circle_index=None,
        max_bounces=3,
    )
    logger.info("Multi-obstacle path: %s", path_multi)
    assert len(path_multi) >= 2

    # ------------------------------------------------------------------
    # Predictive multi-step collision core tests
    # ------------------------------------------------------------------
    table = Rectangle(0.0, 0.0, 800.0, 400.0)
    cue = Circle(120.0, 200.0, 20.0)
    target = Circle(400.0, 220.0, 20.0)

    # Glancing aim (slightly above target center line)
    pred = compute_collision_prediction(
        cue=cue,
        target=target,
        boundary=table,
        aim_direction=(1.0, 0.12),
        max_bounces=3,
    )
    render_array = pred.to_render_array()
    logger.info("Predictive collide=%s impact=%s", pred.did_collide, pred.impact_point)
    logger.info("Render array names: %s", [r["name"] for r in render_array])
    for entry in render_array:
        logger.info("  %s nodes (%d): %s", entry["name"], len(entry["nodes"]), entry["nodes"])

    assert pred.did_collide, "Expected interception with target radius boundary"
    assert pred.impact_point is not None
    assert len(render_array) == 3
    assert [r["name"] for r in render_array] == ["initial", "target", "deflection"]

    # Initial ray stops at impact (exactly 2 nodes: A → intercept)
    assert len(pred.initial_ray) == 2
    assert pred.initial_ray[0] == cue.center
    assert pred.initial_ray[-1] == pred.impact_point

    # Impact lies on target's radius boundary
    dist_to_b = math.hypot(
        pred.impact_point[0] - target.cx, pred.impact_point[1] - target.cy
    )
    assert abs(dist_to_b - target.radius) < 1.5, (
        f"Impact not on B radius boundary: dist={dist_to_b}, r={target.radius}"
    )

    # Both secondary rays start at the split/impact point
    assert pred.target_ray[0] == pred.impact_point
    assert pred.deflection_ray[0] == pred.impact_point
    assert len(pred.target_ray) >= 2
    assert len(pred.deflection_ray) >= 1
    assert len(pred.combined_nodes) == sum(len(r.nodes) for r in pred.combined_rays)

    # Head-on collision: deflection should collapse (no tangent component)
    head_on = compute_collision_prediction(
        cue=Circle(100.0, 200.0, 20.0),
        target=Circle(400.0, 200.0, 20.0),
        boundary=table,
        aim_direction=(1.0, 0.0),
        max_bounces=3,
    )
    logger.info(
        "Head-on: target_nodes=%d deflection_nodes=%d",
        len(head_on.target_ray),
        len(head_on.deflection_ray),
    )
    assert head_on.did_collide
    assert len(head_on.target_ray) >= 2
    # Pure head-on → deflection ray is impact-only (or extremely short)
    assert len(head_on.deflection_ray) <= 2

    # Miss: aim away from target → initial ray only (with possible wall bounces)
    miss = compute_collision_prediction(
        cue=Circle(100.0, 50.0, 20.0),
        target=Circle(400.0, 350.0, 20.0),
        boundary=table,
        aim_direction=(1.0, 0.0),
        max_bounces=3,
    )
    logger.info("Miss collide=%s rays=%s", miss.did_collide, [r.name for r in miss.combined_rays])
    assert not miss.did_collide
    assert len(miss.combined_rays) == 1
    assert miss.combined_rays[0].name == "initial"

    # Wall bounce appears on a secondary ray aimed at a nearby wall
    near_wall = compute_collision_prediction(
        cue=Circle(100.0, 200.0, 15.0),
        target=Circle(200.0, 40.0, 15.0),
        boundary=table,
        aim_direction=None,  # aim directly at target
        max_bounces=3,
    )
    logger.info(
        "Near-wall target path (%d): %s",
        len(near_wall.target_ray),
        near_wall.target_ray,
    )
    assert near_wall.did_collide
    assert len(near_wall.target_ray) >= 2

    logger.info("Self-test PASSED")
    # Return combined predictive nodes as the primary deliverable
    return pred.combined_nodes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen-capture shape tracker with recursive ray-tracing physics",
    )
    parser.add_argument("--left", type=int, default=0, help="Capture region left (px)")
    parser.add_argument("--top", type=int, default=0, help="Capture region top (px)")
    parser.add_argument("--width", type=int, default=1280, help="Capture region width")
    parser.add_argument("--height", type=int, default=720, help="Capture region height")
    parser.add_argument("--fps", type=float, default=60.0, help="Target capture FPS")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (default: run until quit)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable OpenCV preview window",
    )
    parser.add_argument(
        "--dir-x",
        type=float,
        default=1.0,
        help="Default ray direction X (used when no motion detected)",
    )
    parser.add_argument(
        "--dir-y",
        type=float,
        default=0.0,
        help="Default ray direction Y (used when no motion detected)",
    )
    parser.add_argument(
        "--circle-index",
        type=int,
        default=0,
        help="Cue circle index (object A)",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=1,
        help="Target circle index (object B)",
    )
    parser.add_argument(
        "--max-bounces",
        type=int,
        default=3,
        help="Max consecutive wall bounces (capped at 3)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline synthetic validation and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.self_test:
        path = run_self_test()
        print("PATH_NODES =", path)
        return 0

    if args.width <= 0 or args.height <= 0:
        logger.error("Capture width/height must be positive")
        return 2

    capture_cfg = CaptureConfig(
        left=args.left,
        top=args.top,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
    )
    detection_cfg = DetectionConfig(max_wall_bounces=min(max(args.max_bounces, 0), 3))

    tracker = ShapeTracker(
        capture_config=capture_cfg,
        detection_config=detection_cfg,
        visualize=not args.no_visualize,
        ray_direction=(args.dir_x, args.dir_y),
        tracked_circle_index=args.circle_index,
        target_circle_index=args.target_index,
    )

    try:
        path = tracker.run(max_frames=args.max_frames)
    except Exception as exc:
        logger.exception("Tracker failed: %s", exc)
        return 1

    print("PATH_NODES =", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
