"""Streamlit human adjudication UI — 2D V/A slider annotation.

실행:
  streamlit run model/autoEQ/pseudo_label/human_ui/app.py -- \\
    --base_dir dataset/autoEQ/CCMovies

컬럼 가정:
  - labels/test_gold_queue.csv 또는 labels/disagreement_queue.csv
  - windows/<film_id>/<window_id>.mp4
  - emofilm_annotations/va_<film>.csv (선택)

저장: labels/human_annotations.csv (append-only).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st


def parse_cli() -> dict:
    """Streamlit `--` 뒤에 붙은 args 파싱. Streamlit이 `--`를 소비하고 남은 토큰만 sys.argv로 전달."""
    argv = sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=Path, required=True)
    args = p.parse_args(argv)
    return {"base_dir": args.base_dir.resolve()}


@st.cache_resource
def get_paths(base_dir: Path) -> dict:
    return {
        "base_dir": base_dir,
        "gold_queue": base_dir / "labels" / "test_gold_queue.csv",
        "dis_queue": base_dir / "labels" / "disagreement_queue.csv",
        "final_labels": base_dir / "labels" / "final_labels.csv",
        "windows_dir": base_dir / "windows",
        "emofilm_dir": base_dir / "emofilm_annotations",
    }


def load_queue(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_window_mp4(windows_dir: Path, film_id: str, window_id: str) -> Path | None:
    candidate = windows_dir / film_id / f"{window_id}.mp4"
    return candidate if candidate.is_file() else None


def load_emofilm_series(emofilm_dir: Path, film_id: str):
    # EMOFILM_MAP 복사 (순환 import 방지)
    mapping = {
        "big_buck_bunny": "va_bigbuckbunny.csv",
        "sintel": "va_sintel.csv",
        "tears_of_steel": "va_tearsofsteel.csv",
    }
    name = mapping.get(film_id)
    if not name:
        return None
    path = emofilm_dir / name
    if not path.is_file():
        return None
    return pd.read_csv(path)


def render_annotate_tab(paths: dict):
    # sys.path 보정 (streamlit run은 cwd를 기준으로 함)
    repo_root = paths["base_dir"].parents[2] if len(paths["base_dir"].parents) >= 3 else Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from model.autoEQ.pseudo_label.human_ui.annotation_io import (
        append_annotation, progress_by_evaluator, read_all,
    )

    st.sidebar.header("평가자")
    evaluator_id = st.sidebar.text_input("Evaluator ID", value="self")

    queue_choice = st.sidebar.radio(
        "Queue 선택",
        ["test_gold_queue", "disagreement_queue"],
        index=0,
    )
    queue_path = (
        paths["gold_queue"] if queue_choice == "test_gold_queue" else paths["dis_queue"]
    )
    queue_df = load_queue(queue_path)
    if queue_df.empty:
        st.warning(f"{queue_path.name} 가 없거나 비어 있습니다. build_dataset.py 먼저 실행.")
        return

    st.sidebar.caption(f"Queue size: {len(queue_df)}")

    # 진행률 계산
    all_ann = read_all(paths["base_dir"])
    prog = progress_by_evaluator(all_ann)
    done_n = prog.get(evaluator_id, 0)
    st.sidebar.metric(f"{evaluator_id} 완료", f"{done_n} windows")

    # 미평가 window만 필터
    if not all_ann.empty:
        done_ids = set(
            all_ann[all_ann["evaluator_id"] == evaluator_id]["window_id"].astype(str)
        )
    else:
        done_ids = set()
    remaining = queue_df[~queue_df["window_id"].astype(str).isin(done_ids)]
    st.sidebar.caption(f"남은 windows: {len(remaining)}")

    if remaining.empty:
        st.success(f"이 queue({queue_choice})는 {evaluator_id}가 모두 평가 완료.")
        return

    # 현재 index 관리
    row = remaining.iloc[0]
    film_id = str(row["film_id"])
    window_id = str(row["window_id"])

    st.header(f"{film_id} — {window_id}")
    t0 = row.get("t0", None)
    t1 = row.get("t1", None)
    if t0 is not None and t1 is not None and not pd.isna(t0):
        st.caption(f"time: {t0:.1f}s ~ {t1:.1f}s")

    mp4 = find_window_mp4(paths["windows_dir"], film_id, window_id)
    if mp4:
        st.video(str(mp4))
    else:
        st.error(f"window mp4 not found: {paths['windows_dir']}/{film_id}/{window_id}.mp4")

    # 모델 제안값 (참고용 — annotation bias 우려 시 숨김 가능)
    show_hints = st.sidebar.checkbox("모델 제안값 표시", value=True)
    if show_hints:
        cols = st.columns(3)
        if "ensemble_v" in row:
            cols[0].metric("Ensemble V/A",
                           f"{row.get('ensemble_v', 0):.2f} / {row.get('ensemble_a', 0):.2f}")
        if "gemini_v" in row and not pd.isna(row.get("gemini_v")):
            cols[1].metric("Gemini V/A",
                           f"{row.get('gemini_v'):.2f} / {row.get('gemini_a'):.2f}")
        if "quadrant" in row:
            cols[2].metric("Ensemble Quadrant", str(row["quadrant"]))

    # Emo-FilM 1Hz V/A 라인 차트 (매칭 영화만)
    series = load_emofilm_series(paths["emofilm_dir"], film_id)
    if series is not None and t0 is not None and t1 is not None:
        window_ser = series[(series["t_sec"] >= t0 - 10) & (series["t_sec"] <= t1 + 10)]
        if not window_ser.empty:
            st.caption("Emo-FilM 1Hz V/A (참고)")
            st.line_chart(
                window_ser.set_index("t_sec")[["valence", "arousal"]]
            )

    # V/A slider
    st.subheader("Valence / Arousal 평가")
    v = st.slider("Valence", -1.0, 1.0, 0.0, 0.05,
                  help="-1 = 불쾌/슬픔, 0 = 중립, +1 = 즐거움/쾌적")
    a = st.slider("Arousal", -1.0, 1.0, 0.0, 0.05,
                  help="-1 = 차분/평온, 0 = 중립, +1 = 격렬/긴장")
    notes = st.text_input("메모 (선택)", value="")

    if "start_ts" not in st.session_state or st.session_state.get("last_window") != window_id:
        st.session_state["start_ts"] = time.time()
        st.session_state["last_window"] = window_id

    if st.button("💾 저장 & 다음", type="primary"):
        dur = time.time() - st.session_state["start_ts"]
        append_annotation(
            base_dir=paths["base_dir"],
            evaluator_id=evaluator_id,
            window_id=window_id,
            v=v, a=a,
            notes=notes,
            duration_sec=dur,
        )
        st.session_state["start_ts"] = time.time()
        st.rerun()


def render_stats_tab(paths: dict):
    repo_root = paths["base_dir"].parents[2] if len(paths["base_dir"].parents) >= 3 else Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from model.autoEQ.pseudo_label.human_ui.annotation_io import (
        read_all, latest_per_evaluator, aggregate_ratings, progress_by_evaluator,
    )
    from model.autoEQ.pseudo_label.human_ui.krippendorff import (
        krippendorff_alpha_interval,
    )

    df = read_all(paths["base_dir"])
    st.subheader("평가자별 진행률")
    prog = progress_by_evaluator(df)
    if prog:
        st.bar_chart(pd.Series(prog, name="windows"))
    else:
        st.info("아직 annotation 없음")

    # Krippendorff α (2+ evaluators)
    latest = latest_per_evaluator(df)
    n_eval = latest["evaluator_id"].nunique() if not latest.empty else 0
    st.subheader(f"Krippendorff's α (평가자 수: {n_eval})")
    if n_eval >= 2:
        agg = aggregate_ratings(df)
        v_ratings = {w: vals["v"] for w, vals in agg.items() if len(vals["v"]) >= 2}
        a_ratings = {w: vals["a"] for w, vals in agg.items() if len(vals["a"]) >= 2}
        alpha_v = krippendorff_alpha_interval(v_ratings)
        alpha_a = krippendorff_alpha_interval(a_ratings)
        cols = st.columns(2)
        cols[0].metric("α (Valence)", f"{alpha_v:.3f}" if alpha_v == alpha_v else "N/A")
        cols[1].metric("α (Arousal)", f"{alpha_a:.3f}" if alpha_a == alpha_a else "N/A")
        st.caption("목표: 0.7 이상 (interval metric 기준)")
    else:
        st.info("2명 이상 평가자 필요 — 현재 1명만 있음")

    st.subheader("최근 annotations")
    if not df.empty:
        st.dataframe(df.tail(20))


def main():
    params = parse_cli()
    base_dir: Path = params["base_dir"]
    st.set_page_config(page_title="MoodEQ Human Annotation", layout="wide")
    st.title("🎬 MoodEQ — Human V/A Adjudication")
    st.caption(f"base_dir: `{base_dir}`")

    paths = get_paths(base_dir)

    tab_annotate, tab_stats = st.tabs(["✍️ Annotate", "📊 Stats"])
    with tab_annotate:
        render_annotate_tab(paths)
    with tab_stats:
        render_stats_tab(paths)


if __name__ == "__main__":
    main()
