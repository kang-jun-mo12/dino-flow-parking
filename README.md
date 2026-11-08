# DINO Flow Parking

DINOv2 기반 DDP 앞단에 Flow Matching 플래너를 붙인 CARLA 엔드투엔드 주차 모델과, 이 모델을 잰 DDP 평가 코드입니다.

Dino-Diffusion Parking(DDP)의 인식 앞단을 고정해 Fused BEV까지 만들고, Diffusion 플래너 자리를 Flow 플래너로 바꿨습니다. 플래너는 기어 구조를 먼저 고른 뒤 다음 16스텝(약 6.4초)의 곡률·속도 스플라인을 생성하고, 1.6초마다 다시 계획합니다.

```text
카메라 6대 → DINOv2 ViT-B/14 reg → Lift & Splat → BEV 인코더 → 목표 FiLM → Fused BEV (267ch, 16×16)   [고정]
                                                                              │
             목표 포즈 + 자차 상태 → 구조 헤드(F, F→R, R, R→F) → Flow Matching → 16스텝 경로 → pure pursuit
```

## 결과

CARLA 0.9.11, Town04_Opt, 슬롯 16개 × 시작 위치 6개 × 4회, 날씨별 384회입니다. 체크포인트는 `weights/sq30b_s1.pt`입니다.

| 날씨 | 회수 | TSR | 충돌 | 시간 초과 | APE (m) | AOE (°) | APT (s) |
|---|---|---|---|---|---|---|---|
| Same | 384 | 1.000 | 0 | 0 | 0.242 | 1.28 | 16.2 |
| Mild | 384 | 1.000 | 0 | 0 | 0.264 | 1.31 | 16.4 |
| Large | 384 | 1.000 | 0 | 0 | 0.262 | 1.35 | 16.5 |

회차별 기록은 `results/`에 있고, 아래 명령으로 위 표를 다시 만듭니다.

```bash
python tools/summarize_bench.py results/same.csv results/mild.csv results/large.csv
```

이 수치를 읽을 때 알아둘 점입니다.

- **위 표는 Windows DX 렌더로 잰 값입니다.** 이 평가는 OpenGL로 돌려야 합니다(아래 [렌더러](#렌더러-opengl-필수)). 표는 OpenGL 필수 조건을 정하기 전에 잰 것입니다.
- 재계획 주기와 pure pursuit 추적기는 방법의 일부로 DDP와 다릅니다.
- 평가 슬롯은 학습에서 대부분 제외된 슬롯입니다. 800개 학습 궤적 중 20개만 이 슬롯에 있습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `scripts/structure_flow/model.py` | Flow 플래너 |
| `scripts/structure_flow/train.py` | 캐시된 Fused BEV로 학습 |
| `scripts/structure_flow/labels.py`, `trim_labels.py` | 라벨 생성, 경로 끝을 슬롯 중심 0.30 m 앞으로 앞당기는 라벨 수정 |
| `scripts/structure_flow/plan_server.py` | 평가용 플래너 프로세스 (Python 3.11) |
| `scripts/structure_flow/benchmark.py` | DDP 평가 프로토콜 (Python 3.7, CARLA 0.9.11) |
| `scripts/structure_flow/wsl_carla.sh` | Windows에서 WSL2로 OpenGL 서버를 띄우는 스크립트 |
| `scripts/ddp_frontend/` | `last.ckpt`에 맞춰 다시 만든 DDP 앞단, 카메라 리그, BEV 캐시 |
| `scripts/sync_flow/`, `scripts/k4_flow_solver/` | 플래너가 재사용하는 BEV 토큰, 좌표 변환, U-Net 블록 |
| `tools/` | 앞단 가중치 변환, 결과 집계, 오프라인 평가 |
| `weights/sq30b_s1.pt` | 학습된 Flow 플래너 (11 MB) |
| `results/` | 날씨별 `episodes` 기록과 학습 로그 |
| `docs/DESIGN_KR.md` | 설계 기록. 결과 표는 작성 당시 값입니다. |

## 평가 실행

CARLA 클라이언트 egg가 Python 3.7 전용이라 프로세스 두 개로 나뉩니다.

서버와 `CARLA_EGG`는 모두 CARLA 0.9.11이어야 합니다. egg 파일이 없거나 버전이 다르면 `benchmark.py`가 시작하지 않습니다.

### 렌더러: OpenGL 필수

CARLA 서버는 반드시 `-opengl`로 띄웁니다. DDP와 E2E Parking 공식 README가 모두 Linux `./CarlaUE4.sh -opengl`로 서버를 띄웁니다. Windows 기본값인 DX로 띄우면 같은 장면도 카메라 영상이 달라집니다.

렌더러에 따라 결과가 실제로 바뀝니다. E2E Parking 공개 체크포인트를 같은 장면 96개의 첫 프레임에 넣어 비교했습니다.

| 비교 | 영상 PSNR | 첫 기어 선택 일치 |
|---|---|---|
| DX vs DX (같은 렌더러 두 번) | 56.4 dB | 95% |
| DX vs OpenGL | 25.8 dB | 52% |

같은 회차 46개를 실제로 주행시키면 성공률이 DX 45.7%, OpenGL 69.6%였습니다.

`benchmark.py`는 서버의 렌더러를 확인하지 못합니다. 서버 실행 명령에 `-opengl`이 있는지 직접 확인하세요.

**Linux.** 서버를 이렇게 띄우고, 평가 쪽 `CARLA_EGG`에는 `PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg`를 넣습니다. 나머지 순서는 아래 2~4단계와 같습니다. 네이티브 Linux에서는 직접 시험하지 않았습니다.

```bash
./CarlaUE4.sh -opengl -quality-level=Epic -windowed -ResX=640 -ResY=360
```

**Windows.** CARLA 0.9.11의 Windows판은 `-opengl`에서 크래시합니다. 대신 WSL2 안에서 Linux판 CARLA를 띄우고, 플래너와 평가는 Windows에서 그대로 돌립니다. WSL은 `localhost` 포트를 Windows로 넘겨 주므로 연결 설정은 바꿀 필요가 없습니다. 아래는 Windows 11, WSL2 Ubuntu 24.04, RTX 5070 Ti에서 확인한 순서입니다.

1. WSL Ubuntu에서 root로 Linux판 CARLA 0.9.11(4.3 GB)을 받고, 서버를 돌릴 일반 사용자를 만듭니다. UE4는 root로 실행되지 않습니다.

   ```bash
   apt-get update && apt-get install -y mesa-utils
   useradd -m carla
   mkdir -p /home/carla/carla0911
   curl -L -o CARLA_0.9.11.tar.gz https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.11.tar.gz
   tar -xzf CARLA_0.9.11.tar.gz -C /home/carla/carla0911
   chown -R carla:carla /home/carla/carla0911
   ```

2. GPU로 OpenGL이 되는지 확인합니다. `D3D12 (NVIDIA ...)`가 나와야 합니다. `llvmpipe`가 나오면 CPU 렌더라 쓸 수 없습니다.

   ```bash
   GALLIUM_DRIVER=d3d12 glxinfo -B | grep "renderer string"
   ```

3. 저장소의 `wsl_carla.sh`로 서버를 띄웁니다. Mesa의 d3d12 드라이버를 쓰고, Mesa가 UE4용으로 두는 `allow_glsl_cross_stage_interpolation_mismatch` 옵션을 켭니다. 이 옵션이 없으면 Town04_Opt를 불러올 때 `Failed to link program`으로 크래시합니다. 인자는 포트이고 기본값은 2000입니다. WSL에서는 Windows의 저장소가 `/mnt/c/...` 아래에 보이므로 그 경로에서 실행합니다.

   ```bash
   bash scripts/structure_flow/wsl_carla.sh 2000
   ```

4. Windows에서 아래 3~4단계대로 플래너와 평가를 돌립니다.

WSL의 OpenGL은 Mesa가 D3D12로 바꿔 그립니다. 엔진의 OpenGL 셰이더 경로는 Linux와 같지만 드라이버 층은 다릅니다. 이 변환이 CPU 스레드 하나에 몰려 DX보다 약 5배 느리고, GPU 사용률은 5% 안팎입니다.

서버를 여러 개 띄워 슬롯을 나누면 전체가 빨라집니다. 실측으로 서버 5개가 1개의 약 3.1배였습니다. 서버마다 포트를 10씩 띄우고(2000, 2010, ...), 플래너도 서버마다 하나씩 띄웁니다(`plan_server.py --port 6100`, 6101, ...). 평가는 `--port`, `--server-port`, `--slots`, `--out`을 나눠 줍니다. 서버는 40초쯤 간격을 두고 차례로 띄우세요. 동시에 지도를 불러오면 `D3D12: Removing Device`로 서버가 죽을 수 있습니다. 나눠 돌린 `episodes.csv`는 머리줄 하나로 합친 뒤 집계합니다.

### 실행 순서

**1. 앞단 가중치를 받습니다.** 434 MB라 git이 아니라 [Release](https://github.com/kang-jun-mo12/dino-flow-parking/releases/tag/v1.0)에 있습니다.

```bash
curl -L -o weights/ddp_frontend.pt https://github.com/kang-jun-mo12/dino-flow-parking/releases/download/v1.0/ddp_frontend.pt
```

SHA-256은 `556bab3150757fe01c11f2d4cf0030183620727f545a1c2386cbfcc4738dfdd5`입니다. [Hugging Face](https://huggingface.co/ChampagneAndfragrance/Dino_Diffusion_Parking)의 `last.ckpt`에서 `python tools/export_ddp_frontend.py path/to/last.ckpt`로 직접 만들어도 같은 텐서가 나옵니다.

**2. CARLA 0.9.11 서버를 OpenGL로 켭니다.** Linux는 `./CarlaUE4.sh -opengl ...`, Windows는 WSL에서 `wsl_carla.sh`입니다(위 [렌더러](#렌더러-opengl-필수)).

**3. 플래너를 켭니다.** (Python 3.11, CUDA)

```bash
python scripts/structure_flow/plan_server.py weights/sq30b_s1.pt
```

**4. 평가를 돌립니다.** (Python 3.7) `CARLA_EGG`에 0.9.11 egg 경로를 넣습니다. 날씨는 `same`, `mild`, `large`입니다. `requirements.txt`는 플래너용입니다. 평가 쪽 Python 3.7에는 아래 두 패키지만 설치합니다. `setuptools`는 egg를 여는 데 필요합니다.

```bash
python -m pip install numpy setuptools
```

PowerShell:

```powershell
$env:CARLA_EGG = "path\to\carla-0.9.11-py3.7-win-amd64.egg"
python scripts/structure_flow/benchmark.py --out bench/same --weather same --epochs 4
```

Git Bash:

```bash
CARLA_EGG=path/to/carla-0.9.11-py3.7-win-amd64.egg python scripts/structure_flow/benchmark.py --out bench/same --weather same --epochs 4
```

## 학습

학습 데이터는 저장소에 없습니다. Hugging Face의 `ai4ce-drive/CarlaParking`(`parkinglot_1000`, 승인 필요)과 `Jas000n/e2e_dataset`을 받아 `scripts/ddp_frontend/build_cache.py`로 Fused BEV를 캐시합니다. 학습 궤적은 800개입니다.

`weights/sq30b_s1.pt`는 세 단계로 만들었습니다. 첫 단계로 캐시를 묶고, 둘째 단계로 라벨을 고치고, 셋째 단계로 시드 1, 100에폭을 학습합니다. 검증 점수가 가장 좋은 75에폭이 저장됐습니다.

```bash
python scripts/structure_flow/train.py E2E_DATASET PARKINGLOT_1000 --out runs/base --train-tasks 800
```

```bash
python scripts/structure_flow/trim_labels.py PACK runs/pack_sq30b --roots E2E_DATASET PARKINGLOT_1000 --mode squeeze --stop-depth -0.30
```

```bash
python scripts/structure_flow/train.py --pack runs/pack_sq30b --out runs/sq30b_s1 --epochs 100 --seed 1
```

`PACK`은 첫 단계가 만드는 `runs/base/pack_<해시>` 폴더입니다.

## 출처와 사용 조건

이 저장소에는 아직 라이선스를 지정하지 않았습니다.

- **DDP:** 앞단 구조와 `benchmark.py`의 평가 프로토콜은 Dino-Diffusion Parking 공개 코드([`ChampagneAndfragrance/Anonomous_DDP`](https://github.com/ChampagneAndfragrance/Anonomous_DDP))를 따릅니다. 이 코드 저장소에는 라이선스 파일이 없으므로 외부 공개 전에 저자 확인이 필요합니다. Release의 `ddp_frontend.pt`는 MIT로 표시된 Hugging Face 모델 `ChampagneAndfragrance/Dino_Diffusion_Parking`의 `last.ckpt`에서 인식 모듈만 뽑은 것이고, 학습은 저자들이 했습니다.
- **DINOv2:** 앞단의 백본이며 `ddp_frontend.pt`에 가중치가 포함됩니다. Apache-2.0입니다([`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2)).
- **CAA-Policy:** 카메라 리그 설정의 출처입니다([`ai4ce/CAAPolicy`](https://github.com/ai4ce/CAAPolicy)).
