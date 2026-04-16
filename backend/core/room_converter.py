"""
Phase 2.2: RoomPlan JSON → PyRoomAcoustics 입력 형식 변환.

좌표계: RoomPlan(y-up) → PRA(z-up), (x, y, z) → (x, -z, y).
바닥 다각형: 벽 하단 모서리들의 convex hull.
가구: 표면적 가중으로 벽 평균 흡음률에 가산 (MVP 근사).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import Point, Polygon

from core.absorption_db import HybridAbsorptionDatabase, PRA_COMPATIBLE_BANDS_HZ

logger = logging.getLogger(__name__)


class InsufficientWallsError(ValueError):
    pass


class InvalidRoomGeometryError(ValueError):
    pass


@dataclass
class ConvertedObject:
    id: str
    category: str
    position: np.ndarray
    dimensions: np.ndarray
    rotation_z_deg: float
    absorption: List[float]
    confidence: str


@dataclass
class ConvertedRoom:
    floor_corners_2d: np.ndarray
    height: float
    wall_avg_absorption: List[float]
    floor_absorption: List[float]
    ceiling_absorption: List[float]
    listener_position: np.ndarray
    objects: List[ConvertedObject] = field(default_factory=list)
    room_bbox: Dict[str, float] = field(default_factory=dict)

    @property
    def floor_area_m2(self) -> float:
        x = self.floor_corners_2d[:, 0]
        y = self.floor_corners_2d[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def volume_m3(self) -> float:
        return self.floor_area_m2 * self.height


def roomplan_to_pra_coords(p: np.ndarray) -> np.ndarray:
    """(x, y, z)_RoomPlan → (x, -z, y)_PRA."""
    return np.array([p[0], -p[2], p[1]], dtype=float)


def _parse_transform(matrix16: List[float]) -> np.ndarray:
    if matrix16 is None or len(matrix16) != 16:
        raise InvalidRoomGeometryError("transform은 16개 float 배열이어야 합니다")
    return np.array(matrix16, dtype=float).reshape(4, 4, order="F")


def _transform_center(matrix16: List[float]) -> np.ndarray:
    m = _parse_transform(matrix16)
    return m[:3, 3].copy()


def _rotation_y_rad(matrix16: List[float]) -> float:
    m = _parse_transform(matrix16)
    return float(np.arctan2(m[0, 2], m[2, 2]))


def extract_wall_bottom_edge(wall: Dict[str, Any]) -> np.ndarray:
    """벽 하단 2D 좌표 2점 (PRA 평면 x-y). shape=(2, 2)."""
    m = _parse_transform(wall["transform"])
    center = m[:3, 3]
    dims = wall["dimensions"]
    half_w = float(dims[0]) * 0.5
    half_h = float(dims[1]) * 0.5
    local_pts = np.array([
        [-half_w, -half_h, 0.0, 1.0],
        [+half_w, -half_h, 0.0, 1.0],
    ])
    world_pts = (m @ local_pts.T).T[:, :3]
    result = np.zeros((2, 2), dtype=float)
    for i, wp in enumerate(world_pts):
        pra_pt = roomplan_to_pra_coords(wp)
        result[i, 0] = pra_pt[0]
        result[i, 1] = pra_pt[1]
    return result


def compute_floor_polygon(walls: List[Dict[str, Any]]) -> np.ndarray:
    if len(walls) < 3:
        raise InsufficientWallsError(
            f"최소 3개의 벽이 필요합니다 (입력: {len(walls)}개)"
        )
    pts: List[np.ndarray] = []
    for w in walls:
        try:
            edge = extract_wall_bottom_edge(w)
            pts.append(edge[0])
            pts.append(edge[1])
        except InvalidRoomGeometryError as e:
            logger.warning("벽 %s 파싱 실패, 스킵: %s", w.get("id"), e)

    if len(pts) < 3:
        raise InsufficientWallsError("유효한 벽 모서리가 부족합니다")

    points = np.array(pts)
    try:
        hull = ConvexHull(points)
    except QhullError as e:
        raise InvalidRoomGeometryError(f"Convex hull 계산 실패: {e}")

    corners = points[hull.vertices]
    if _polygon_area_signed(corners) < 0:
        corners = corners[::-1]
    return corners


def _polygon_area_signed(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_room_height(walls: List[Dict[str, Any]]) -> float:
    heights = [float(w["dimensions"][1]) for w in walls if len(w.get("dimensions", [])) >= 2]
    if not heights:
        return 2.7
    return float(np.mean(heights))


def _convert_object(
    obj: Dict[str, Any],
    db: HybridAbsorptionDatabase,
    material_key: Optional[str] = None,
) -> Optional[ConvertedObject]:
    try:
        matrix = _parse_transform(obj["transform"])
    except InvalidRoomGeometryError as e:
        logger.warning("object %s transform 파싱 실패: %s", obj.get("id"), e)
        return None

    center_rp = matrix[:3, 3]
    pos_pra = roomplan_to_pra_coords(center_rp)
    dims = np.array(obj["dimensions"], dtype=float)
    rot = _rotation_y_rad(obj["transform"])
    rot_deg = float(np.degrees(rot))

    absorption = db.get_absorption_array(obj.get("category", "unknown"), material_key)
    return ConvertedObject(
        id=obj.get("id", ""),
        category=obj.get("category", "unknown"),
        position=pos_pra,
        dimensions=dims,
        rotation_z_deg=rot_deg,
        absorption=absorption,
        confidence=obj.get("confidence", "unknown"),
    )


def _weighted_wall_absorption(
    wall_abs: List[float],
    walls: List[Dict[str, Any]],
    room_height: float,
    converted_objects: List[ConvertedObject],
) -> List[float]:
    wall_surface_area = sum(
        float(w["dimensions"][0]) * room_height for w in walls
    ) or 1.0

    obj_surface = 0.0
    obj_weighted_abs = np.zeros(6, dtype=float)
    for o in converted_objects:
        w, d, h = float(o.dimensions[0]), float(o.dimensions[1]), float(o.dimensions[2])
        surf = 2.0 * (w * d + w * h + d * h)
        obj_surface += surf
        obj_weighted_abs += np.array(o.absorption[:6], dtype=float) * surf

    total = wall_surface_area + obj_surface
    wall_weighted = np.array(wall_abs[:6], dtype=float) * wall_surface_area
    combined = (wall_weighted + obj_weighted_abs) / total
    return [float(min(0.95, max(0.01, c))) for c in combined]


def convert_roomplan_to_pra_input(
    scan_json: Dict[str, Any],
    db: HybridAbsorptionDatabase,
    listener_height: float = 1.2,
    user_material_selections: Optional[Dict[str, str]] = None,
) -> ConvertedRoom:
    walls = scan_json.get("walls", []) or []
    objects_raw = scan_json.get("objects", []) or []

    floor_corners = compute_floor_polygon(walls)
    height = compute_room_height(walls)

    selections = user_material_selections or {}
    converted_objects: List[ConvertedObject] = []
    for obj in objects_raw:
        mat_key = selections.get(obj.get("id"))
        co = _convert_object(obj, db, mat_key)
        if co is not None:
            converted_objects.append(co)

    wall_abs = db.get_absorption_array("wall")
    floor_abs = db.get_absorption_array("floor")
    ceiling_abs = db.get_absorption_array("ceiling")
    wall_abs_weighted = _weighted_wall_absorption(
        wall_abs, walls, height, converted_objects
    )

    xs = floor_corners[:, 0]
    ys = floor_corners[:, 1]
    bbox = {
        "x_min": float(xs.min()), "x_max": float(xs.max()),
        "y_min": float(ys.min()), "y_max": float(ys.max()),
        "z_min": 0.0, "z_max": float(height),
    }

    poly = Polygon(floor_corners.tolist())
    origin_2d = Point(0.0, 0.0)
    if poly.contains(origin_2d):
        listener_xy = np.array([0.0, 0.0])
    else:
        logger.info("스캔 원점이 바닥 다각형 외부 — centroid를 청취자 위치로 사용")
        centroid = poly.centroid
        listener_xy = np.array([centroid.x, centroid.y])
    listener = np.array(
        [float(listener_xy[0]), float(listener_xy[1]), float(listener_height)],
        dtype=float,
    )

    return ConvertedRoom(
        floor_corners_2d=floor_corners,
        height=float(height),
        wall_avg_absorption=wall_abs_weighted,
        floor_absorption=floor_abs,
        ceiling_absorption=ceiling_abs,
        listener_position=listener,
        objects=converted_objects,
        room_bbox=bbox,
    )
