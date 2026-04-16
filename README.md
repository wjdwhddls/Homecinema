포빅아 ai 프로젝트

---

## Windows 환경 주의사항

Windows에서 개발/실행할 때 아래 4가지를 꼭 지켜주세요. 한글 경로 + OneDrive 조합에서는 일부 ML 라이브러리가 **침묵 속에 실패**할 수 있어 디버깅이 까다롭습니다.

### 1. 프로젝트는 영문 경로 + OneDrive 바깥에 두기

- **권장**: `C:\dev\mood-eq` 처럼 한글·공백이 없고 **OneDrive 동기화 폴더 바깥**에 있는 짧은 경로
- **피할 것**: `C:\Users\<이름>\OneDrive\바탕 화면\...` 같은 한글 포함 경로, 특히 OneDrive 동기화 폴더
- **이유**: Silero VAD(`torch.jit.load`) 등 일부 PyTorch C++ 런타임이 내부적으로 Windows ANSI(cp949) 기반 `fopen`을 호출합니다. UTF-8로 표현된 한글 경로를 디코딩하지 못해 다음과 같이 실패합니다.
  ```
  RuntimeError: open file failed because of errno 42 on fopen:
  Illegal byte sequence, file path: C:\...\바탕 화면\...\silero_vad.jit
  ```
- **OneDrive 우회 불가**: `GetShortPathNameW`로 얻은 8.3 short path조차 OneDrive 하위 폴더에서는 한글이 유지되는 경우가 많아 우회가 어렵습니다. 아예 영문 경로로 이동하는 것이 깔끔합니다.

### 2. Python 실행 전 UTF-8 환경변수 설정

Python / pip / 워커 / 백엔드 명령 앞에 다음 두 줄을 매번 export 하세요 (Git Bash 기준).

```bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
```

- `PYTHONIOENCODING=utf-8` — stdout/stderr 한글 출력 시 `UnicodeEncodeError: 'cp949' codec can't encode ...` 방지
- `PYTHONUTF8=1` — subprocess(`ffprobe` 등) 호출의 기본 디코딩을 UTF-8로 강제. 설정하지 않으면 자식 프로세스의 UTF-8 출력을 cp949로 디코딩하려다 실패해 `result.stdout`이 `None`이 되고 `json.loads(None) → TypeError`로 이어집니다.

### 3. 백엔드는 `uvicorn` 직접 호출로 실행

`backend/main.py`의 `if __name__ == "__main__":` 블록은 `uvicorn.run(..., reload=True)`를 사용하는데, Windows에서는 WatchFiles가 재시작을 반복하며 서버가 즉시 종료되는 현상이 확인되었습니다. 대신 `uvicorn`을 직접 호출하세요.

```bash
cd backend
source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
uvicorn main:app --host 0.0.0.0 --port 8000
```

코드 변경 시에는 수동으로 `Ctrl+C` → 재실행합니다. `--reload`가 필요한 경우에는 별도 환경(WSL2/Linux)에서 돌리는 편이 안정적입니다.

### 4. 권장 셸: Git Bash

이 문서와 워커 가이드의 명령 예시는 **Git Bash** 기준입니다. PowerShell이나 cmd에서도 동작은 가능하지만 환경변수 설정·경로 구분자·활성화 스크립트 문법이 달라서 매번 변환이 필요합니다.

| 작업 | Git Bash | PowerShell | cmd |
|---|---|---|---|
| venv 활성화 | `source venv/Scripts/activate` | `venv\Scripts\Activate.ps1` | `venv\Scripts\activate.bat` |
| 환경변수 | `export VAR=value` | `$env:VAR = "value"` | `set VAR=value` |
| 현재 디렉토리 참조 | `$(pwd)` | `$PWD` | `%cd%` |
| 파일 다운로드 | `curl -L -o file URL` | `Invoke-WebRequest -OutFile file URL` | `curl.exe -L -o file URL` |

Git Bash를 쓰면 README의 명령을 복붙해도 그대로 동작합니다. 특히 `export VAR=value && ...` 같은 체인과 `$(pwd)` 치환이 Unix 방식이라 편합니다.
