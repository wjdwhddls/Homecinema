"""
하이브리드 흡음률 DB 조회 모듈 (Phase 2.1).

`backend/data/absorption_db_v1.1.json`을 로드해 PyRoomAcoustics Material 객체로
변환한다. `pra_available=True`이면 공식 키워드를, 아니면 custom 배열을 사용한다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyroomacoustics as pra

logger = logging.getLogger(__name__)

PRA_COMPATIBLE_BANDS_HZ: List[int] = [125, 250, 500, 1000, 2000, 4000]


@dataclass
class AbsorptionEntry:
    category: str
    material_key: Optional[str]
    pra_keyword: Optional[str]
    pra_available: bool
    absorption_7band: List[float]
    source: str
    confidence: str
    notes: str
    section: str


class HybridAbsorptionDatabase:
    _DEFAULT_FILENAMES = ("absorption_db_v1.1.json", "absorption_db.json")
    _FALLBACK_CATEGORY = "wall"

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or self._auto_detect_path()
        with open(self._path, "r", encoding="utf-8") as f:
            self._db: Dict[str, Any] = json.load(f)
        self._surfaces: Dict[str, Any] = self._db.get("surfaces", {})
        self._objects: Dict[str, Any] = self._db.get("objects", {})
        logger.info("AbsorptionDB 로드 완료: %s", self._path)

    @staticmethod
    def _auto_detect_path() -> Path:
        candidates: List[Path] = []
        here = Path(__file__).resolve().parent.parent
        for name in HybridAbsorptionDatabase._DEFAULT_FILENAMES:
            candidates.append(here / "data" / name)
            candidates.append(Path.cwd() / "data" / name)
            candidates.append(Path.cwd() / name)
        for p in candidates:
            if p.is_file():
                return p
        raise FileNotFoundError(
            f"absorption DB를 찾을 수 없습니다. 다음 경로 중 하나에 배치: "
            + ", ".join(str(c) for c in candidates[:3])
        )

    def get_entry(
        self, category: str, material: Optional[str] = None
    ) -> AbsorptionEntry:
        section: str
        raw: Optional[Dict[str, Any]]
        if category in self._surfaces:
            section = "surfaces"
            raw = self._surfaces[category]
        elif category in self._objects:
            section = "objects"
            raw = self._objects[category]
        else:
            logger.warning(
                "알 수 없는 카테고리 '%s' — 폴백: objects.unknown 또는 surfaces.wall",
                category,
            )
            if "unknown" in self._objects:
                section = "objects"
                raw = self._objects["unknown"]
                category = "unknown"
            else:
                section = "surfaces"
                raw = self._surfaces[self._FALLBACK_CATEGORY]
                category = self._FALLBACK_CATEGORY

        if material and "alternatives" in raw and material in raw["alternatives"]:
            alt = raw["alternatives"][material]
            return AbsorptionEntry(
                category=category,
                material_key=material,
                pra_keyword=alt.get("pra_keyword"),
                pra_available=bool(alt.get("pra_available", False)),
                absorption_7band=list(alt.get("absorption_7band", [])),
                source=alt.get("source", ""),
                confidence=alt.get("confidence", raw.get("confidence", "low")),
                notes=alt.get("description", alt.get("notes", "")),
                section=section,
            )

        return AbsorptionEntry(
            category=category,
            material_key=None,
            pra_keyword=raw.get("pra_keyword"),
            pra_available=bool(raw.get("pra_available", False)),
            absorption_7band=list(raw.get("absorption_7band", [])),
            source=raw.get("source", ""),
            confidence=raw.get("confidence", "low"),
            notes=raw.get("notes", ""),
            section=section,
        )

    def make_pra_material(
        self, category: str, material: Optional[str] = None
    ) -> pra.Material:
        entry = self.get_entry(category, material)
        return self._entry_to_material(entry)

    def _entry_to_material(self, entry: AbsorptionEntry) -> pra.Material:
        coeffs = entry.absorption_7band[:6] if entry.absorption_7band else [0.1] * 6
        if len(coeffs) < 6:
            coeffs = coeffs + [coeffs[-1]] * (6 - len(coeffs))

        if entry.pra_available and entry.pra_keyword:
            try:
                return pra.Material(energy_absorption=entry.pra_keyword)
            except (KeyError, ValueError) as e:
                logger.warning(
                    "PRA 키워드 '%s' 로드 실패 (%s) — custom 배열로 폴백",
                    entry.pra_keyword, e,
                )
        return pra.Material(
            energy_absorption={
                "coeffs": coeffs,
                "center_freqs": PRA_COMPATIBLE_BANDS_HZ,
            }
        )

    def make_room_materials(
        self,
        scan_json: Dict[str, Any],
        user_selections: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pra.Material]:
        """방 구성요소별 Material 딕셔너리. 키는 'wall'/'floor'/'ceiling' +
        각 object의 id. user_selections은 object_id → material_key."""
        materials: Dict[str, pra.Material] = {}
        materials["wall"] = self.make_pra_material("wall")
        materials["floor"] = self.make_pra_material("floor")
        materials["ceiling"] = self.make_pra_material("ceiling")
        selections = user_selections or {}
        for obj in scan_json.get("objects", []) or []:
            oid = obj.get("id")
            cat = obj.get("category", "unknown")
            mat_key = selections.get(oid)
            materials[oid] = self.make_pra_material(cat, mat_key)
        return materials

    def get_absorption_array(
        self, category: str, material: Optional[str] = None
    ) -> List[float]:
        """6밴드 흡음률 배열 (125~4000Hz). 시뮬레이션 직접 주입용."""
        entry = self.get_entry(category, material)
        coeffs = list(entry.absorption_7band[:6])
        if len(coeffs) < 6:
            coeffs = coeffs + [coeffs[-1] if coeffs else 0.1] * (6 - len(coeffs))
        return coeffs

    def get_provenance(
        self, category: str, material: Optional[str] = None
    ) -> Dict[str, str]:
        entry = self.get_entry(category, material)
        return {
            "category": entry.category,
            "material": entry.material_key or "default",
            "source": entry.source,
            "confidence": entry.confidence,
            "notes": entry.notes,
        }
