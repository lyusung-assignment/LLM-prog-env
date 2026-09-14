# llm-docker

상명대 LLM Programming Lab1 환경(Python 3.11 / CUDA 12.6 / PyTorch 2.13 / Transformers 5.14 / PEFT·TRL / LangGraph / Jupyter)을 **Docker 컨테이너**로 재현한 버전입니다.
버전은 수업 레퍼런스 장비(`Lab1_VSCode.pdf`)에 맞춰져 있습니다.
Anaconda를 호스트에 직접 까는 대신 이미지 안에 Miniconda로 conda 환경 `llm`을 만들어 두어, 데스크탑을 갈아엎어도 `git clone` → `docker compose up` 몇 줄로 복구됩니다.

대상 환경: **Windows 데스크탑 + Docker Desktop(WSL2 백엔드) + NVIDIA GPU**

---

## 1. 데스크탑 최초 준비 (1회만)

1. **NVIDIA 드라이버** 최신 버전 설치 (CUDA 12.x 마이너 버전 호환 덕분에 Windows 드라이버 527.41 이상이면 12.6 컨테이너도 동작하지만, 되도록 최신 드라이버를 쓰세요)
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
- **JupyterLab**: http://localhost:8888 → 토큰은 `.env`의 `JUPYTER_TOKEN` (`.env`를 안 만들었으면 compose 기본값 `llm`이 쓰입니다)
- **TensorBoard**: 컨테이너 안에서 `tensorboard --logdir <경로> --host 0.0.0.0` 으로 띄우면 http://localhost:6006 으로 붙습니다
- 셸 진입: `docker compose exec llm bash` → 프롬프트가 `(llm)`으로 뜹니다
- 종료: `docker compose down` / 재시작: `docker compose up -d`
  `docker compose down -v` 는 `hf-cache` 볼륨까지 지웁니다. 모델 가중치를 다시 받게 되니 주의하세요.

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

## 5. Lab별 환경 만들기

Lab마다 요구하는 Python/PyTorch 버전이 다를 때 씁니다. 메인 이미지의 `llm` 환경은 공통 베이스로 그대로 두고, lab별 환경만 따로 만듭니다.

```
workspace/lab01/
├─ environment.yml   # 이 lab이 요구하는 스펙 (python/torch/transformers 버전)
└─ check_env.py      # 스펙이 맞는지 확인하는 스크립트
```

만들기 (최초 1회, 수 분 걸립니다):

```
docker compose exec llm conda env create -f /workspace/lab01/environment.yml -p /envs/lab01
```

Jupyter 커널로 등록 (노트북에서 쓰려면):

```
docker compose exec llm conda run -p /envs/lab01 python -m ipykernel install --user --name lab01 --display-name "Python (lab01)"
```

스펙 확인:

```
docker compose exec llm conda run -p /envs/lab01 --no-capture-output python /workspace/lab01/check_env.py
```

그 환경으로 코드 실행:

```
docker compose exec llm conda run -p /envs/lab01 --no-capture-output python /workspace/lab01/main.py
```

삭제:

```
docker compose exec llm conda env remove -p /envs/lab01
```

- **`/envs` 와 커널 등록은 볼륨입니다.** 여기 만든 환경은 `docker compose down` 을 해도 남습니다. 단 `docker compose down -v` 는 지웁니다.
- 새 lab을 시작할 땐 `workspace/lab01` 폴더를 복사한 뒤 `environment.yml` 의 `name:` 과 버전만 고치면 됩니다.
- 이미지에 구운 `llm` 환경은 `/opt/conda/envs/llm` 에 그대로 있습니다. 볼륨이 가리지 않습니다.
- `-p /envs/lab01` 대신 `-n lab01` 로 써도 `envs_dirs` 최우선이 `/envs` 라 같은 자리에 생깁니다. 경로를 쓰는 편이 헷갈리지 않습니다.
- **환경 하나당 6GB 안팎입니다** (torch + CUDA 라이브러리). lab 사이에 버전 차이가 없다면 굳이 나누지 말고 공통 `llm` 환경을 쓰세요.

## 6. SSH로 상태만 확인하기

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

> Windows OpenSSH 서버의 기본 셸은 **`cmd.exe`** 입니다. 위 명령들은 cmd에서도 그대로 동작하지만, PowerShell 문법(`Copy-Item` 등)을 쓰고 싶으면 `ssh desktop powershell -c "..."` 처럼 명시하거나 서버의 기본 셸을 바꾸세요:
>
> ```
> New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
>   -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force
> ```

## 7. 패키지 추가

- **임시로 써보기**: `docker compose exec llm pip install <패키지>` — 컨테이너를 지우면 사라집니다.
- **확정**: `Dockerfile`의 해당 `pip install` 블록에 줄을 추가하고 `docker compose build`. 아래쪽 블록에 추가할수록 재빌드가 빠릅니다.
- `conda install`은 피하세요. PyTorch를 pip 휠(cu126)로 깔았기 때문에 conda가 의존성을 풀면서 torch를 CPU 버전으로 바꿔버리는 일이 생깁니다.
- OS 레벨 패키지(`ffmpeg` 등)는 Dockerfile 위쪽 `apt-get install` 블록에 추가합니다.

## 8. 버전을 바꿔야 할 때

버전 핀은 **`Dockerfile` 과 `constraints.txt` 두 곳에 같이** 적혀 있습니다. 한쪽만 고치면 `-c` 제약에 걸려 빌드가 `ResolutionImpossible` 로 즉시 죽습니다 (조용히 틀린 버전이 깔리는 것보다 낫습니다).

CUDA 버전을 바꿀 땐 아래를 함께 맞춰야 합니다. 베이스 이미지 / 휠 인덱스 / torch 버전이 어긋나면 빌드는 통과하는데 `torch.cuda.is_available()` 이 `False` 로 나옵니다.

| 위치 | 현재 값 |
|---|---|
| `Dockerfile` 의 `FROM` | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` |
| `Dockerfile` 의 PyTorch `--index-url` | `.../whl/cu126` |
| `Dockerfile` + `constraints.txt` 의 `torch` / `torchvision` | `2.13.0` / `0.28.0` |
| `constraints.txt` 의 나머지 핀 | transformers 5.14.1, datasets 5.0.0, accelerate 1.14.0 등 |
| `docker-compose.yml` 의 `image` 태그 (표시용) | `llm-lab:cu126` |

`README` 1단계의 GPU 패스스루 확인 명령에 있는 `nvidia/cuda:12.6.3-base-ubuntu22.04` 태그도 같이 바꿔 두면 헷갈리지 않습니다.

`torchaudio` 는 일부러 빠져 있습니다. cu126 인덱스의 최신이 2.11.0 이라 torch 2.13.0 과 짝이 없고, 레퍼런스 장비에도 없습니다.

특정 lab만 다른 버전이 필요하다면 이 파일들을 고치지 말고 **5장의 lab별 환경**을 쓰세요. 메인 이미지는 공통 베이스로 두는 편이 낫습니다.

이미지 크기를 줄이려면 베이스 태그를 `devel` → `runtime`으로 바꾸면 됩니다. 단 `flash-attn` 등을 직접 컴파일할 계획이면 `devel`이 필요합니다(nvcc 포함).

## 9. 문제가 생기면

| 증상 | 원인 / 조치 |
|---|---|
| `torch.cuda.is_available()` 이 False | 1단계 GPU 패스스루 확인부터 다시. Docker Desktop이 WSL2 백엔드인지 점검 |
| 빌드 중 `No matching distribution found` | 고정한 버전이 존재하지 않는 경우. 해당 줄의 `==버전` 을 지우고 재시도 |
| 학습 중 컨테이너가 조용히 죽음 | 공유 메모리 부족. 단 현재 `docker-compose.yml` 에 `ipc: host` 가 있어 호스트의 `/dev/shm` 을 그대로 쓰므로 **`shm_size` 는 무시됩니다.** `ipc: host` 줄을 지워 `shm_size: "16gb"` 를 살리거나, WSL2 쪽 `/dev/shm` 크기를 늘리세요 |
| 8888 접속 시 토큰을 계속 물어봄 | `.env` 가 없어도 기본 토큰 `llm` 이 적용됩니다. 값을 바꾼 뒤 `docker compose up -d` 로 재생성하지 않았거나, 토큰을 잊은 경우 — `docker compose logs llm` 에 찍힌 접속 URL의 `?token=...` 을 쓰세요 |
| 포트 충돌 | `docker-compose.yml` 의 `"8888:8888"` 을 `"8899:8888"` 처럼 앞 숫자만 변경 |
