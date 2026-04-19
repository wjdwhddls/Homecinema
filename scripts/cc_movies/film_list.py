"""CC 영화 인벤토리 — Emo-FilM 14편 + Blender 추가 단편.

각 엔트리는 `FilmEntry` 로 표현되며 다음 정보를 보관한다:
  - film_id: 영화 고유 ID (디스크/메타데이터 키)
  - title: 제목
  - duration_sec: 예상 러닝타임(논문/공식 기재)
  - license: "CC-BY" / "CC-BY-NC" / "CC-BY-SA" / "CC0" / "CC-BY-NC-ND" 등
  - source: "blender" / "vimeo" / "archive" / "youtube"
  - download_url: 실다운 URL (확정된 것만). None이면 수동 확보 필요.
  - emo_film: Emo-FilM 논문에서 쓴 14편에 속하는지 여부

다운로드가 확정되지 않은 엔트리는 `download_url=None` + `notes`에 원저작 크레딧/탐색 힌트 기록.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FilmEntry:
    film_id: str
    title: str
    duration_sec: int
    license: str
    source: str
    download_url: Optional[str]
    emo_film: bool = False
    notes: str = ""


# Emo-FilM 14편 (Morgenroth et al. 2025 Nature Scientific Data). Duration은 논문 Table 1 기재.
EMO_FILM_ENTRIES: list[FilmEntry] = [
    FilmEntry(
        film_id="big_buck_bunny",
        title="Big Buck Bunny",
        duration_sec=490,  # 8:10
        license="CC-BY",
        source="archive",
        download_url="https://archive.org/download/BigBuckBunny_328/BigBuckBunny_512kb.mp4",
        emo_film=True,
    ),
    FilmEntry(
        film_id="sintel",
        title="Sintel",
        duration_sec=722,  # 12:02
        license="CC-BY",
        source="archive",
        download_url="https://archive.org/download/Sintel_201709/Sintel.mp4",
        emo_film=True,
    ),
    FilmEntry(
        film_id="tears_of_steel",
        title="Tears of Steel",
        duration_sec=588,  # 9:48
        license="CC-BY",
        source="archive",
        download_url="https://archive.org/download/Tears-of-Steel/tears_of_steel_1080p.mp4",
        emo_film=True,
    ),
    FilmEntry(
        film_id="the_secret_number",
        title="The Secret Number",
        duration_sec=784,  # 13:04
        license="CC-BY-NC",  # Colin Levy, CC BY-NC 3.0 Unported
        source="vimeo",
        download_url=None,  # Vimeo 43732205. yt-dlp 필요.
        emo_film=True,
        notes="Colin Levy, SCAD 2012. vimeo.com/43732205. CC BY-NC 3.0.",
    ),
    FilmEntry(
        film_id="payload",
        title="Payload",
        duration_sec=1008,  # 16:48
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="Stuart Willis, VCA 2011. payloadfilm.com. CC 라이선스 여부 추가 확인 필요.",
    ),
    FilmEntry(
        film_id="after_the_rain",
        title="After The Rain",
        duration_sec=496,  # 8:16
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="제목 동명 작품 다수. Emo-FilM/LIRIS 출처의 정확한 감독·연도 미확정.",
    ),
    FilmEntry(
        film_id="between_viewings",
        title="Between Viewings",
        duration_sec=808,  # 13:28
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="부동산 에이전트 스토리라인 단편. 직접 URL 미확인.",
    ),
    FilmEntry(
        film_id="chatter",
        title="Chatter",
        duration_sec=405,  # 6:45
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="온라인 호러/고립 스릴러 단편. 직접 URL 미확인.",
    ),
    FilmEntry(
        film_id="first_bite",
        title="First Bite",
        duration_sec=599,  # 9:59
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="Emo-FilM Table 1 romance/teen 단편. 동명 2018 Vincenzo Nappi 호러와 구분 필요.",
    ),
    FilmEntry(
        film_id="lesson_learned",
        title="Lesson Learned",
        duration_sec=667,  # 11:07
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="갱단 폭력 드라마. 직접 URL 미확인.",
    ),
    FilmEntry(
        film_id="spaceman",
        title="Spaceman",
        duration_sec=805,  # 13:25
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="Romance/drama 단편. 동명 여러 작품 있음 — 연도 확인 필요.",
    ),
    FilmEntry(
        film_id="superhero",
        title="Superhero",
        duration_sec=1028,  # 17:08
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="말기암 caregiving 드라마. 직접 URL 미확인.",
    ),
    FilmEntry(
        film_id="to_claire_from_sonny",
        title="To Claire From Sonny",
        duration_sec=402,  # 6:42
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="편지 형식 드라마. 직접 URL 미확인.",
    ),
    FilmEntry(
        film_id="you_again",
        title="You Again",
        duration_sec=798,  # 13:18
        license="UNKNOWN",
        source="vimeo",
        download_url=None,
        emo_film=True,
        notes="고등학교 동창회 로맨스/드라마. 동명 2010 할리우드 장편과 구분.",
    ),
]


# Blender 추가 단편 (Emo-FilM 외). 코퍼스 확장용.
BLENDER_EXTRA_ENTRIES: list[FilmEntry] = [
    FilmEntry(
        film_id="elephants_dream",
        title="Elephants Dream",
        duration_sec=660,  # 11:00
        license="CC-BY",
        source="archive",
        download_url="https://archive.org/download/ElephantsDream/ed_1024.mp4",
    ),
    FilmEntry(
        film_id="cosmos_laundromat",
        title="Cosmos Laundromat: First Cycle",
        duration_sec=731,  # 12:11
        license="CC-BY",
        source="archive",
        download_url="https://archive.org/download/CosmosLaundromatFirstCycle/Cosmos%20Laundromat%20-%20First%20Cycle%20%281080p%29.mp4",
    ),
    FilmEntry(
        film_id="spring",
        title="Spring",
        duration_sec=465,  # 7:45
        license="CC-BY",
        source="archive",
        download_url="https://archive.org/download/springopenmovie/springopenmovie.mp4",
    ),
    FilmEntry(
        film_id="agent_327",
        title="Agent 327: Operation Barbershop",
        duration_sec=230,  # 3:50
        license="CC-BY-ND",  # 공식: CC-BY-ND per Blender Studio
        source="archive",
        download_url="https://archive.org/download/agent327operationbarbershop/agent327-1080.mp4",
    ),
    FilmEntry(
        film_id="caminandes_3",
        title="Caminandes 3: Llamigos",
        duration_sec=152,  # 2:32
        license="CC-BY",
        source="archive",
        download_url="https://archive.org/download/CaminandesLlamigos/Caminandes_%20Llamigos-1080p.mp4",
    ),
    # Hero는 archive.org에 직접 MP4 없음 — YouTube/Blender Cloud 경유. URL TODO.
    FilmEntry(
        film_id="hero",
        title="Hero",
        duration_sec=235,  # 3:55
        license="CC-BY",
        source="youtube",
        download_url=None,
        notes="Daniel M. Lara 2018. Blender Cloud / YouTube. yt-dlp 필요. youtu.be/pKmSdY56VtY?",
    ),
]


# Live-action 확장 (2026-04-19 추가): 애니메이션 편향 완화 목적.
# 각 영화 full feature 중 runtime 25%/50%/75% 지점의 4분씩 × 3 segment = 12분 highlight 사용.
# 원본 mp4는 films/_sources/에 백업, highlight만 films/에 노출.
LIVE_ACTION_HIGHLIGHTS: list[FilmEntry] = [
    FilmEntry(
        film_id="his_girl_friday_highlight",
        title="His Girl Friday (1940) - highlights",
        duration_sec=720,
        license="Public Domain",
        source="archive",
        download_url="https://archive.org/download/his-girl-friday-1940_202109/His%20Girl%20Friday%20%281940%29.ia.mp4",
        notes="Howard Hawks 1940 스크루볼 코미디. 공공영역. 25%/50%/75% × 4min highlights.",
    ),
    FilmEntry(
        film_id="doa_highlight",
        title="D.O.A. (1950) - highlights",
        duration_sec=720,
        license="Public Domain",
        source="archive",
        download_url="https://archive.org/download/d.-o.-a.-with-edmond-o-brien-1950-1080p-hd-film/D.%20O.%20A.%20with%20Edmond%20O%27Brien%201950%20-%201080p%20HD%20Film.mp4",
        notes="Rudolph Maté 1950 film noir. 공공영역. 25%/50%/75% × 4min highlights.",
    ),
    FilmEntry(
        film_id="valkaama_highlight",
        title="Valkaama (2010) - highlights",
        duration_sec=720,
        license="CC-BY-SA-3.0",
        source="archive",
        download_url="https://archive.org/download/PA6827366/PA_682_7366.mp4",
        notes="Tim Baumann 2010 open source drama. CC-BY-SA 3.0 (파생물도 SA). 25%/50%/75% × 4min highlights.",
    ),
]


ALL_FILMS: list[FilmEntry] = EMO_FILM_ENTRIES + BLENDER_EXTRA_ENTRIES + LIVE_ACTION_HIGHLIGHTS


def get_downloadable_films() -> list[FilmEntry]:
    """`download_url`이 확정된 영화만 반환."""
    return [f for f in ALL_FILMS if f.download_url]


def get_pending_films() -> list[FilmEntry]:
    """URL 확정 필요한 영화."""
    return [f for f in ALL_FILMS if f.download_url is None]


def total_downloadable_duration_sec() -> int:
    return sum(f.duration_sec for f in get_downloadable_films())


def summary() -> dict:
    dl = get_downloadable_films()
    pending = get_pending_films()
    total_sec = sum(f.duration_sec for f in ALL_FILMS)
    dl_sec = total_downloadable_duration_sec()
    return {
        "total_films": len(ALL_FILMS),
        "downloadable": len(dl),
        "pending_url": len(pending),
        "total_duration_sec": total_sec,
        "downloadable_duration_sec": dl_sec,
        "windows_at_4s_stride": dl_sec // 4,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    print("\n=== Downloadable films ===")
    for f in get_downloadable_films():
        print(f"  [{f.film_id}] {f.title} ({f.duration_sec}s, {f.license}, {f.source})")
    print("\n=== Pending films (URL 확보 필요) ===")
    for f in get_pending_films():
        print(f"  [{f.film_id}] {f.title} — {f.notes or 'no notes'}")
