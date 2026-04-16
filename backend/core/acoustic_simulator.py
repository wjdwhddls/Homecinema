"""
Phase 3.1: PyRoomAcoustics 기반 음향 시뮬레이션.

주어진 스피커 위치에서 청취자까지의 임펄스 응답(RIR)을 계산하고
RT60, 정재파 심각도, 주파수 평탄도 등의 지표를 산출한다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pyroomacoustics as pra
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import linregress

from core.room_converter import ConvertedRoom

logger = logging.getLogger(__name__)

PRA_BANDS_HZ: List[int] = [125, 250, 500, 1000, 2000, 4000]


class AcousticSimulator:
    def __init__(self, room: ConvertedRoom, sample_rate: int = 16000) -> None:
        self.room = room
        self.fs = sample_rate

    def _make_material(self, coeffs: List[float]) -> pra.Material:
        coeffs6 = list(coeffs[:6])
        if len(coeffs6) < 6:
            coeffs6 += [coeffs6[-1] if coeffs6 else 0.1] * (6 - len(coeffs6))
        return pra.Material(
            energy_absorption={"coeffs": coeffs6, "center_freqs": PRA_BANDS_HZ}
        )

    def build_pra_room(
        self,
        max_order: int = 4,
        ray_tracing: bool = True,
        n_rays: int = 3000,
    ) -> pra.Room:
        corners = self.room.floor_corners_2d.T
        wall_mat = self._make_material(self.room.wall_avg_absorption)
        floor_mat = self._make_material(self.room.floor_absorption)
        ceiling_mat = self._make_material(self.room.ceiling_absorption)

        r = pra.Room.from_corners(
            corners,
            fs=self.fs,
            materials=wall_mat,
            max_order=max_order,
            ray_tracing=ray_tracing,
            air_absorption=True,
        )
        r.extrude(
            self.room.height,
            materials={"floor": floor_mat, "ceiling": ceiling_mat},
        )
        if ray_tracing:
            r.set_ray_tracing(
                n_rays=n_rays,
                receiver_radius=0.3,
                energy_thres=1e-7,
                time_thres=1.0,
            )
        r.add_microphone(self.room.listener_position.tolist())
        return r

    def simulate(
        self,
        speaker_position: np.ndarray,
        max_order: int = 4,
        ray_tracing: bool = True,
    ) -> Optional[np.ndarray]:
        try:
            r = self.build_pra_room(max_order=max_order, ray_tracing=ray_tracing)
            r.add_source(speaker_position.tolist())
            r.compute_rir()
            rir = r.rir[0][0]
            return np.asarray(rir, dtype=float)
        except Exception as e:
            logger.warning("simulate 실패 (pos=%s): %s", speaker_position, e)
            return None

    def compute_metrics(self, rir: np.ndarray) -> Dict[str, float]:
        rt60 = self._estimate_rt60(rir)
        rt60_low = self._estimate_rt60_band(rir, 125, 250)
        rt60_mid = self._estimate_rt60_band(rir, 500, 2000)
        standing = self._standing_wave_severity_db(rir)
        flat = self._flatness_db(rir)
        er_ratio = self._early_reflection_ratio(rir)
        dr_ratio = self._direct_to_reverb_ratio_db(rir)
        return {
            "rt60_seconds": float(rt60),
            "rt60_low": float(rt60_low),
            "rt60_mid": float(rt60_mid),
            "standing_wave_severity_db": float(standing),
            "flatness_db": float(flat),
            "early_reflection_ratio": float(er_ratio),
            "direct_to_reverb_ratio_db": float(dr_ratio),
        }

    def _estimate_rt60(self, rir: np.ndarray) -> float:
        rir_sq = rir.astype(float) ** 2
        if rir_sq.sum() <= 0:
            return 0.0
        edc = np.flipud(np.cumsum(np.flipud(rir_sq)))
        edc = edc / edc[0]
        edc_db = 10.0 * np.log10(edc + 1e-20)

        t = np.arange(len(edc_db)) / self.fs
        mask = (edc_db <= -5.0) & (edc_db >= -35.0)
        if mask.sum() < 10:
            return 0.0
        slope, intercept, *_ = linregress(t[mask], edc_db[mask])
        if slope >= 0:
            return 0.0
        rt60 = -60.0 / slope
        return float(np.clip(rt60, 0.0, 5.0))

    def _bandpass(
        self, rir: np.ndarray, low_hz: float, high_hz: float
    ) -> np.ndarray:
        nyq = self.fs / 2.0
        low = max(low_hz, 20.0) / nyq
        high = min(high_hz, nyq - 1) / nyq
        if low >= high:
            return rir
        sos = butter(4, [low, high], btype="band", output="sos")
        try:
            return sosfiltfilt(sos, rir)
        except ValueError:
            return rir

    def _estimate_rt60_band(
        self, rir: np.ndarray, low_hz: float, high_hz: float
    ) -> float:
        filtered = self._bandpass(rir, low_hz, high_hz)
        return self._estimate_rt60(filtered)

    def _magnitude_db(self, rir: np.ndarray, n_fft: int = 2048) -> tuple:
        if len(rir) < 8:
            return np.array([]), np.array([])
        freqs, psd = welch(rir, fs=self.fs, nperseg=min(n_fft, len(rir)))
        mag_db = 10.0 * np.log10(psd + 1e-20)
        return freqs, mag_db

    def _standing_wave_severity_db(self, rir: np.ndarray) -> float:
        freqs, mag_db = self._magnitude_db(rir)
        if len(freqs) == 0:
            return 0.0
        mask = (freqs >= 20.0) & (freqs <= 250.0)
        if mask.sum() < 3:
            return 0.0
        return float(np.std(mag_db[mask]))

    def _flatness_db(self, rir: np.ndarray) -> float:
        freqs, mag_db = self._magnitude_db(rir)
        if len(freqs) == 0:
            return 0.0
        mask = (freqs >= 50.0) & (freqs <= 8000.0)
        if mask.sum() < 3:
            return 0.0
        return float(np.std(mag_db[mask]))

    def _early_reflection_ratio(self, rir: np.ndarray) -> float:
        if len(rir) < 2:
            return 0.0
        n_early = int(0.080 * self.fs)
        total_e = float(np.sum(rir ** 2))
        if total_e <= 0:
            return 0.0
        early_e = float(np.sum(rir[:n_early] ** 2))
        return float(np.clip(early_e / total_e, 0.0, 1.0))

    def _direct_to_reverb_ratio_db(self, rir: np.ndarray) -> float:
        if len(rir) < 2:
            return 0.0
        n_direct = int(0.005 * self.fs)
        direct_e = float(np.sum(rir[:n_direct] ** 2))
        reverb_e = float(np.sum(rir[n_direct:] ** 2))
        if reverb_e <= 0 or direct_e <= 0:
            return 0.0
        return float(10.0 * np.log10(direct_e / reverb_e))

    def compute_frequency_response(
        self, rir: np.ndarray, n_fft: int = 512
    ) -> Dict[str, List[float]]:
        freqs, mag_db = self._magnitude_db(rir, n_fft=n_fft)
        return {
            "freqs_hz": freqs.tolist(),
            "magnitude_db": mag_db.tolist(),
        }
