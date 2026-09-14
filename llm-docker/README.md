# llm-docker

상명대 LLM Programming Lab1 환경(Python 3.11 / CUDA 12.6 / PyTorch 2.6 / HuggingFace / PEFT·TRL / LangGraph / Jupyter)을 **Docker 컨테이너**로 재현한 버전입니다.
Anaconda를 호스트에 직접 까는 대신 이미지 안에 conda 환경 `llm`을 만들어 두어, 데스크탑을 갈아엎어도 `git clone` + `docker compose up` 두 줄로 복구됩니다.

대상 환경: **Windows 데스크탑 + Docker Desktop(WSL2 백엔드) + NVIDIA GPU**

---

## 1. 데스크탑 최초 준비 (1회만)

1. **NVIDIA 드라이버** 최신 버전 설치 (CUDA 12.6 컨테이너는 드라이버 527.41 이상이면 동작)
2. **Git for Windows** 설치 — https://git-scm.com/download/win
3. **Docker Desktop** 설치 — https://www.docker.com/products/docker-desktop
   설치 후 Settings → General → **Use the WSL 2 based engine** 체크 확인
4. GPU 패스스루 확인. PowerShell에서:

   ```
   docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
   ```

   GPU 표가 출력되면 준비 완료입니다. 여기서 실패하면 아래 어떤 것도 되지 않으니 이 단계부터 해결하세요.

## 2. 클론 & 실행

PowerShell에서:

```
git clone https://github.com/lyusung-assignment/LLM-prog-env.git
cd LLM-prog-env\llm-docker
Copy-Item .env.example .env
docker compose build
docker compose up -d
```

- 최초 빌드는 20~40분 걸립니다 (PyTorch 등 다운로드). 두 번째부터는 캐시로 수 초입니다.
- **JupyterLab**: http://localhost:8888 → 토큰은 `.env`의 `JUPYTER_TOKEN` (기본값 `llm`)
- 셸 진입: `docker compose exec llm bash` → 프롬프트가 `(llm)`으로 뜹니다
- 종료: `docker compose down` / 재시작: `docker compose up -d`

## 3. 설치 확인

```
docker compose exec llm python /workspace/check_gpu.py
```

`cuda avail : True`, `cuda ver : 12.6`, GPU 이름이 나오면 정상입니다.

## 4. 작업 방식

- `./workspace` 폴더가 컨테이너의 `/workspace`와 연결됩니다. **코드는 이 폴더에 두세요.** 호스트에서 편집하고 실행만 컨테이너에서 하면 됩니다.
- `workspace` 밖에 만든 파일은 컨테이너를 지우면 같이 사라집니다.
- 모델 가중치는 `hf-cache` 볼륨에 남습니다. 이미지를 다시 빌드해도 재다운로드하지 않습니다.
- VS Code는 데스크탑에서 그냥 켜서 `workspace` 폴더를 열면 됩니다. 컨테이너 안 인터프리터를 직접 쓰고 싶으면 **Dev Containers** 확장 → `Attach to Running Container...` → `llm-lab`, 인터프리터는 `/opt/conda/envs/llm/bin/python`.

## 5. SSH로 상태만 확인하기

실제 작업은 화면 스트리밍으로 하고, 노트북에서는 SSH로 상태만 볼 때 쓰는 명령들입니다. 로그인 없이 한 줄로 던져도 됩니다.

```
ssh desktop "nvidia-smi"
```

```
ssh desktop "docker ps -a --filter name=llm-lab"
```

```
ssh desktop "docker logs --tail 30 llm-lab"
```

> Windows OpenSSH 서버의 기본 셸은 PowerShell입니다. `docker` 명령은 그대로 동작합니다.

## 6. 패키지 추가

- **임시로 써보기**: `docker compose exec llm pip install <패키지>` — 컨테이너를 지우면 사라집니다.
- **확정**: `Dockerfile`의 해당 `pip install` 블록에 줄을 추가하고 `docker compose build`. 아래쪽 블록에 추가할수록 재빌드가 빠릅니다.
- `conda install`은 피하세요. PyTorch를 pip 휠(cu126)로 깔았기 때문에 conda가 의존성을 풀면서 torch를 CPU 버전으로 바꿔버리는 일이 생깁니다.
- OS 레벨 패키지(`ffmpeg` 등)는 Dockerfile 위쪽 `apt-get install` 블록에 추가합니다.

## 7. 버전을 바꿔야 할 때

CUDA 버전은 **세 곳을 반드시 함께** 바꿔야 합니다. 하나만 바꾸면 빌드는 통과하는데 `torch.cuda.is_available()`이 `False`로 나옵니다.

| 위치 | 현재 값 |
|---|---|
| `Dockerfile` 의 `FROM` | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` |
| `Dockerfile` 의 PyTorch `--index-url` | `.../whl/cu126` |
| `docker-compose.yml` 의 `image` 태그 | `llm-lab:cu126` |

PyTorch 2.6은 CUDA 12.6 휠이 나온 첫 버전입니다. 12.4로 내릴 경우 `torch==2.5.1` + `whl/cu124` + 베이스 이미지 `12.4.1-cudnn-devel-ubuntu22.04` 조합으로 함께 내리세요.

이미지 크기를 줄이려면 베이스 태그를 `devel` → `runtime`으로 바꾸면 됩니다. 단 `flash-attn` 등을 직접 컴파일할 계획이면 `devel`이 필요합니다(nvcc 포함).

## 8. 문제가 생기면

| 증상 | 원인 / 조치 |
|---|---|
| `torch.cuda.is_available()` 이 False | 1단계 GPU 패스스루 확인부터 다시. Docker Desktop이 WSL2 백엔드인지 점검 |
| 빌드 중 `No matching distribution found` | 고정한 버전이 존재하지 않는 경우. 해당 줄의 `==버전` 을 지우고 재시도 |
| 학습 중 컨테이너가 조용히 죽음 | 공유 메모리 부족. `docker-compose.yml`의 `shm_size` 를 더 올리세요 |
| 8888 접속 시 토큰을 계속 물어봄 | `.env` 를 만들지 않았거나 `docker compose up -d` 로 재시작하지 않은 경우 |
| 포트 충돌 | `docker-compose.yml` 의 `"8888:8888"` 을 `"8899:8888"` 처럼 앞 숫자만 변경 |
