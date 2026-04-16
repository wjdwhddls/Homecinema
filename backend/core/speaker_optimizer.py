"""
Phase 3.2: 최적 스피커 배치 탐색.

후보 위치를 음향 휴리스틱으로 생성 → coarse-to-fine 2단계 탐색 →
가중 점수로 정렬해 best + top-5 대안을 반환한다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from core.acoustic_simulator import AcousticSimulator
from core.room_converter import ConvertedRoom

logger = logging.getLogger(__name__)

ConfigType = Literal["single", "stereo"]


@dataclass
class Candidate:
    config_type: ConfigType
    left: np.ndarray
    right: np.ndarray

    @property
    def is_stereo(self) -> bool:
        return self.config_type == "stereo"

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "left": self.left.tolist(),
            "right": self.right.tolist(),
        }


@dataclass
class EvaluatedCandidate:
    candidate: Candidate
    score: float
    metrics: Optional[Dict[str, float]]
    rank: int = 0


class SpeakerOptimizer:
    # 점 소스 가정(치수 미입력) 시 기본 여유 마진
    _DEFAULT_WALL_MARGIN_M = 0.30
    _DEFAULT_CORNER_MARGIN_M = 0.80

    def __init__(
        self,
        simulator: AcousticSimulator,
        config_type: ConfigType = "stereo",
        time_budget_seconds: float = 180.0,
        speaker_dimensions: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sim = simulator
        self.config_type = config_type
        self.time_budget = time_budget_seconds
        self._room = simulator.room
        self._poly = Polygon(self._room.floor_corners_2d.tolist())

        # 스피커 치수(meter). 방향 미지이므로 수평면의 가장 큰 축(max(W, D))의
        # 절반을 클리어런스 기준으로 사용해 어느 방향이든 안전하게 놓을 수 있도록 한다.
        if speaker_dimensions is not None:
            spk_w = float(speaker_dimensions.get("width_m", 0.0))
            spk_d = float(speaker_dimensions.get("depth_m", 0.0))
        else:
            spk_w = spk_d = 0.0
        self._spk_half_max = max(spk_w, spk_d) / 2.0

        # 클리어런스는 기본 마진을 하한으로 유지. 큰 스피커일수록 더 떨어져야 함.
        self._wall_margin = max(self._DEFAULT_WALL_MARGIN_M, self._spk_half_max + 0.05)
        self._corner_margin = max(
            self._DEFAULT_CORNER_MARGIN_M, self._spk_half_max + 0.30
        )
        # 청취 최소거리도 스피커 크기에 따라 증가
        self._min_listen_distance = max(1.5, self._spk_half_max + 0.5)

    # ------------------------------------------------------------------
    # 후보 생성
    # ------------------------------------------------------------------
    def estimate_forward_direction(self) -> np.ndarray:
        listener_xy = self._room.listener_position[:2]
        corners = self._room.floor_corners_2d
        n = len(corners)
        best_edge_normal = np.array([1.0, 0.0])
        best_dist = 0.0
        for i in range(n):
            a = corners[i]
            b = corners[(i + 1) % n]
            mid = 0.5 * (a + b)
            d = np.linalg.norm(mid - listener_xy)
            if d > best_dist:
                best_dist = d
                direction = mid - listener_xy
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    best_edge_normal = direction / norm
        return best_edge_normal

    def _generate_stereo_pair(
        self,
        forward: np.ndarray,
        angle_deg: float,
        distance: float,
        height: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        listener = self._room.listener_position
        theta = np.radians(angle_deg)
        rot_l = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        rot_r = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)],
        ])
        dir_l = rot_l @ forward
        dir_r = rot_r @ forward
        left_xy = listener[:2] + dir_l * distance
        right_xy = listener[:2] + dir_r * distance
        left = np.array([left_xy[0], left_xy[1], height], dtype=float)
        right = np.array([right_xy[0], right_xy[1], height], dtype=float)
        return left, right

    def _is_valid_position(self, pos_xy: np.ndarray) -> bool:
        pt = Point(pos_xy[0], pos_xy[1])
        if not self._poly.contains(pt):
            return False
        if pt.distance(self._poly.boundary) < self._wall_margin:
            return False
        corners = self._room.floor_corners_2d
        for c in corners:
            if np.linalg.norm(pos_xy - c) < self._corner_margin:
                return False
        return True

    def generate_candidates(self, n_candidates: int = 30) -> List[Candidate]:
        forward = self.estimate_forward_direction()
        candidates: List[Candidate] = []
        angles = np.linspace(22.5, 37.5, 5)
        dist_lo = max(1.5, self._min_listen_distance)
        dist_hi = max(3.0, dist_lo + 1.0)
        distances = np.linspace(dist_lo, dist_hi, 4)
        heights = np.linspace(1.0, 1.4, 3)

        for angle in angles:
            for dist in distances:
                for h in heights:
                    left, right = self._generate_stereo_pair(forward, angle, dist, h)
                    if not (self._is_valid_position(left[:2])
                            and self._is_valid_position(right[:2])):
                        continue
                    if self.config_type == "stereo":
                        candidates.append(Candidate("stereo", left, right))
                    else:
                        mid = (left + right) * 0.5
                        candidates.append(Candidate("single", mid, mid))
                    if len(candidates) >= n_candidates:
                        return candidates
        return candidates

    def _generate_refined_candidates(
        self, seed: Candidate, n: int = 5
    ) -> List[Candidate]:
        refined: List[Candidate] = []
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, -0.1],
        ])
        for off in offsets[:n]:
            if seed.is_stereo:
                new_l = seed.left + off
                new_r = seed.right - np.array([off[0], off[1], 0.0]) + np.array([0.0, 0.0, off[2]])
            else:
                new_l = seed.left + off
                new_r = new_l
            if seed.is_stereo and not (self._is_valid_position(new_l[:2])
                                       and self._is_valid_position(new_r[:2])):
                continue
            if not seed.is_stereo and not self._is_valid_position(new_l[:2]):
                continue
            refined.append(Candidate(seed.config_type, new_l, new_r))
        return refined

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------
    def evaluate_candidate(
        self, cand: Candidate, max_order: int = 4, ray_tracing: bool = True
    ) -> EvaluatedCandidate:
        if cand.is_stereo:
            rir_l = self.sim.simulate(cand.left, max_order=max_order, ray_tracing=ray_tracing)
            rir_r = self.sim.simulate(cand.right, max_order=max_order, ray_tracing=ray_tracing)
            if rir_l is None or rir_r is None:
                return EvaluatedCandidate(cand, float("inf"), None)
            n = max(len(rir_l), len(rir_r))
            rir_l = np.pad(rir_l, (0, n - len(rir_l)))
            rir_r = np.pad(rir_r, (0, n - len(rir_r)))
            rir_sum = rir_l + rir_r
        else:
            rir = self.sim.simulate(cand.left, max_order=max_order, ray_tracing=ray_tracing)
            if rir is None:
                return EvaluatedCandidate(cand, float("inf"), None)
            rir_sum = rir

        metrics = self.sim.compute_metrics(rir_sum)
        score = (
            3.0 * metrics["standing_wave_severity_db"]
            + 2.0 * metrics["flatness_db"]
            + 10.0 * (1.0 - metrics["early_reflection_ratio"])
            - 0.5 * metrics["direct_to_reverb_ratio_db"]
        )
        if cand.is_stereo:
            dist_l = float(np.linalg.norm(cand.left - self._room.listener_position))
            dist_r = float(np.linalg.norm(cand.right - self._room.listener_position))
            score += 10.0 * abs(dist_l - dist_r)
        return EvaluatedCandidate(cand, float(score), metrics)

    # ------------------------------------------------------------------
    # 최적화
    # ------------------------------------------------------------------
    def find_optimal(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        stage1 = self.generate_candidates(n_candidates=30)
        logger.info("stage1 후보 개수: %d", len(stage1))
        if not stage1:
            raise RuntimeError("유효한 후보 위치를 생성하지 못했습니다")

        stage1_evaluated: List[EvaluatedCandidate] = []
        total = len(stage1)
        for idx, cand in enumerate(stage1):
            if (time.time() - start) > self.time_budget:
                logger.warning("time budget 초과 — stage1 조기 종료")
                break
            ec = self.evaluate_candidate(cand, max_order=3, ray_tracing=True)
            stage1_evaluated.append(ec)
            if progress_callback:
                progress_callback(idx + 1, total)

        stage1_evaluated.sort(key=lambda x: x.score)

        stage2_evaluated: List[EvaluatedCandidate] = []
        top_seeds = [ec for ec in stage1_evaluated[:5] if ec.score < float("inf")]
        for seed in top_seeds:
            if (time.time() - start) > self.time_budget:
                logger.warning("time budget 초과 — stage2 조기 종료")
                break
            refined = self._generate_refined_candidates(seed.candidate, n=5)
            for r in refined:
                if (time.time() - start) > self.time_budget:
                    break
                ec = self.evaluate_candidate(r, max_order=5, ray_tracing=True)
                stage2_evaluated.append(ec)

        combined = stage1_evaluated + stage2_evaluated
        combined.sort(key=lambda x: x.score)
        valid = [c for c in combined if c.score < float("inf") and c.metrics is not None]
        if not valid:
            raise RuntimeError("모든 후보의 평가가 실패했습니다")

        for i, ec in enumerate(valid):
            ec.rank = i + 1

        elapsed = time.time() - start
        return {
            "best": valid[0],
            "top5": valid[:5],
            "all_evaluated": valid,
            "elapsed_seconds": float(elapsed),
            "config_type": self.config_type,
        }
