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

- **DDP 평가기와 남은 차이가 있습니다.** Windows DX 렌더를 썼고(`-opengl`은 Windows에서 크래시), 재계획 주기와 pure pursuit 추적기는 방법의 일부로 DDP와 다릅니다.
- 평가 슬롯은 학습에서 대부분 제외된 슬롯입니다. 800개 학습 궤적 중 20개만 이 슬롯에 있습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `scripts/structure_flow/model.py` | Flow 플래너 |
| `scripts/structure_flow/train.py` | 캐시된 Fused BEV로 학습 |
| `scripts/structure_flow/labels.py`, `trim_labels.py` | 라벨 생성, 경로 끝을 슬롯 중심 0.30 m 앞으로 앞당기는 라벨 수정 |
| `scripts/structure_flow/plan_server.py` | 평가용 플래너 프로세스 (Python 3.11) |
| `scripts/structure_flow/benchmark.py` | DDP 평가 프로토콜 (Python 3.7, CARLA 0.9.11) |
| `scripts/ddp_frontend/` | `last.ckpt`에 맞춰 다시 만든 DDP 앞단, 카메라 리그, BEV 캐시 |
| `scripts/sync_flow/`, `scripts/k4_flow_solver/` | 플래너가 재사용하는 BEV 토큰, 좌표 변환, U-Net 블록 |
| `tools/` | 앞단 가중치 변환, 결과 집계, 오프라인 평가 |
| `weights/sq30b_s1.pt` | 학습된 Flow 플래너 (11 MB) |
| `results/` | 날씨별 `episodes` 기록과 학습 로그 |
| `docs/DESIGN_KR.md` | 설계 기록. 결과 표는 작성 당시 값입니다. |

## 평가 실행

CARLA 클라이언트 egg가 Python 3.7 전용이라 프로세스 두 개로 나뉩니다.

서버와 `CARLA_EGG`는 모두 CARLA 0.9.11이어야 합니다. egg 파일이 없거나 버전이 다르면 `benchmark.py`가 시작하지 않습니다. 위 결과는 Windows의 DX 렌더로 잰 값이라, 다른 OS나 GPU에서는 약간 다를 수 있습니다.

**1. 앞단 가중치를 받습니다.** 434 MB라 git이 아니라 [Release](https://github.com/kang-jun-mo12/dino-flow-parking/releases/tag/v1.0)에 있습니다.

```bash
curl -L -o weights/ddp_frontend.pt https://github.com/kang-jun-mo12/dino-flow-parking/releases/download/v1.0/ddp_frontend.pt
```

SHA-256은 `556bab3150757fe01c11f2d4cf0030183620727f545a1c2386cbfcc4738dfdd5`입니다. [Hugging Face](https://huggingface.co/ChampagneAndfragrance/Dino_Diffusion_Parking)의 `last.ckpt`에서 `python tools/export_ddp_frontend.py path/to/last.ckpt`로 직접 만들어도 같은 텐서가 나옵니다.

**2. CARLA 0.9.11 서버를 켭니다.**

```bash
CarlaUE4.exe -quality-level=Epic -windowed -ResX=640 -ResY=360
```

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
