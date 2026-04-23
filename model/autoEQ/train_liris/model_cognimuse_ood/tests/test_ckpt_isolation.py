"""BASE weights isolation test (CI-level gate).

Phase 4-A 의 어느 스크립트도 `runs/phase2a/2a2_A_K7_s*/best.pt` 또는
`runs/phase3/test_final_metrics.json` 을 변경해서는 안 된다.

이 테스트는 알려진 MD5 체크섬과 대조한다. 체크섬은 2026-04-22 Phase 3
FREEZE 시점의 값이며, 이후 어느 코드도 이 파일을 덮어쓰지 않았음을
증명한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

BASE_WEIGHTS_MD5 = {
    "runs/phase2a/2a2_A_K7_s42/best.pt":
        "27b8a9fc8bcd6b422dbdfca37402c506",
    "runs/phase2a/2a2_A_K7_s123/best.pt":
        "5b64d84ebda610370fb5aaa6aafcaf00",
    "runs/phase2a/2a2_A_K7_s2024/best.pt":
        "9d0a3503784703fa33398f3897d9429e",
}

# Phase 4-A 출력 금지 경로들
FORBIDDEN_OUTPUT_ROOTS = [
    Path("runs/phase2a"),
    Path("runs/phase3"),
]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.parametrize("rel_path,expected", list(BASE_WEIGHTS_MD5.items()))
def test_base_weight_md5_unchanged(rel_path: str, expected: str):
    p = Path(rel_path)
    assert p.is_file(), f"BASE weight missing: {p}"
    actual = _md5(p)
    assert actual == expected, (
        f"BASE weight tampered!\n  path    = {p}\n"
        f"  expected MD5 = {expected}\n  actual MD5   = {actual}"
    )


def test_output_path_assertion_exists():
    """run_eval_ood.py 에 'runs/cognimuse/' prefix assertion 이 실제로 있는지 확인."""
    src = Path("model/autoEQ/train_liris/model_cognimuse_ood/run_eval_ood.py").read_text()
    assert "runs/cognimuse/" in src, "output path prefix assertion missing"
    assert "startswith" in src, "output path prefix assertion missing"


def test_variant_package_does_not_modify_base():
    """Phase 4-A 패키지의 파일이 BASE 파일을 import 하지만 수정하지 않는지 (정적 검사).

    변경 패턴 감지: `cfg.{field} = `, `TrainLirisConfig()` 의 mutate, 등.
    """
    pkg_dir = Path("model/autoEQ/train_liris/model_cognimuse_ood")
    for src_file in pkg_dir.rglob("*.py"):
        # 테스트 파일 자체에는 forbidden pattern 이 문자열로 포함되므로 제외
        if src_file.parent.name == "tests":
            continue
        text = src_file.read_text()
        # BASE module의 TrainLirisConfig defaults 수정 의심 패턴
        # (cfg 인스턴스에 attribute 할당하는 것은 허용 — override 목적이므로)
        # 금지: 모듈 레벨에서 TrainLirisConfig 의 default 필드 변경
        forbidden_patterns = [
            "TrainLirisConfig.lr =",
            "TrainLirisConfig.weight_decay =",
            "TrainLirisConfig.num_mood_classes =",
            "TrainLirisConfig.head_dropout =",
        ]
        for pat in forbidden_patterns:
            assert pat not in text, f"BASE config mutation suspected in {src_file}: {pat}"
