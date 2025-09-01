# 설치 방법

## 0. pyvenv 설정(선택)

```bash
python3 -m venv {{ VENV_NAME }}

# linux의 경우
. {{ VENV_NAME }}/bin/activate

# windows의 경우
.\{{ VENV_NAME }}\Scripts\activate
```

## 1. yolo 설치

```bash
pip install ultralytics
```

## 2. 변환 컨버터 설치

```bash
# 만약 ios 네이티브에서 돌리려면 coremltools도 설치
pip install onnx tensorflow
```

# YOLO 모델 생성

> model별 특징
> | 모델 | 특징 | 속도 | 용도 |
> | ----------------------- | ----------------------- | ----- | ----------------------- |
> | `yolo11n` (nano) | 가장 작은 모델, 매우 빠름, 정확도 낮음 | 초고속 | 임베디드/모바일, 프로토타입 |
> | `yolo11s` (small) | 경량 모델, 속도와 정확도 균형 | 빠름 | 모바일/경량 서버 |
> | `yolo11m` (medium) | 중간 규모, 정확도 ↑ | 중간 | 일반 서버, 데스크탑 |
> | `yolo11l` (large) | 큰 모델, 정확도 높음 | 느림 | 고성능 GPU, 배치 추론 |
> | `yolo11x` (extra-large) | 가장 큰 모델, 최고 정확도 | 매우 느림 | 고급 GPU, 고해상도 이미지, 배치 처리 |

> yolo 11을 사용합니다. yolo 8과 yolo 11이 있는데, 둘 다 flutter에서 지원 될 뿐더러, yolo 11이 파라미터 최적화가 더 되어있습니다.

> yolo의 훈련 / 추론 / 변환 결과는 아래와 같이 구성됩니다. 이때 proejct는 `runs/results`로 전부 동일하다고 가정합니다.
>
> ```bash
> results
> ├── model_1 # 훈련만 완료된 모델
> │   └── weights
> │       ├── best.pt
> │       └── last.pt
> ├── model_2 # tflite 변환 완료된 모델
> │   └── weights
> │       ├── best_saved_model
> │       │   ├── best_float16.tflite
> │       │   └── best_float32.tflite
> │       ├── best.pt
> │       └── last.pt
> ├── predict1 # 추론 결과 1
> └── predict2 # 추론 결과 2, 다른 project 경로를 주지 않는 이상
> ```

## 훈련

```bash
# yolo는 640x640 이미지를 사용한다.
yolo detect train model=yolo11s.pt data=datasets/data.yaml epochs=100 imgsz=640 project=runs/results name=output

# 이미 훈련한 것이 있다면 아래와 같이 훈련한 pt를 넣는다.
yolo detect train model={{ 훈련한 모델 경로 }} data=datasets/data.yaml epochs=100 imgsz=640 project=runs/results name=output
```

## 추론

```bash
yolo detect predict model={{ 훈련한 모델 경로 }} source=datasets/test/images project=runs/results hide_conf=True
```

## 변환

> ios에서 mlcore만 써야 한다 생각하지만, 이미지 검출은 단순해서 cpu만으로도 빠름

```bash
# android / ios
yolo export model={{ 훈련한 모델 경로 }} format=tflite

# ios native
yolo export model={{ 훈련한 모델 경로 }} format=coreml
```

# yolo 파라미터

## 기본

| 파라미터    | 설명                                     | 예시                 |
| ----------- | ---------------------------------------- | -------------------- |
| `model`     | 훈련할 모델 이름/경로 (.pt)              | `yolo11n.pt`         |
| `data`      | 데이터셋 YAML 경로                       | `datasets/data.yaml` |
| `epochs`    | 학습 epoch 수                            | `100`                |
| `batch`     | 배치 사이즈                              | `16`                 |
| `imgsz`     | 입력 이미지 크기 (정사각형)              | `640`                |
| `device`    | 학습 장치 (`cpu` 또는 GPU 번호)          | `0`                  |
| `optimizer` | 옵티마이저 종류 (`SGD`, `Adam`, `AdamW`) | `AdamW`              |

## 학습 최적화

| 파라미터        | 설명                                            |
| --------------- | ----------------------------------------------- |
| `lr0`           | 초기 learning rate                              |
| `momentum`      | SGD 모멘텀                                      |
| `weight_decay`  | L2 정규화                                       |
| `accumulate`    | gradient 누적 횟수 (작은 batch로 큰 batch 효과) |
| `multi_scale`   | 다중 스케일 학습                                |
| `image_weights` | 이미지별 가중치 적용                            |
| `sync_bn`       | 다중 GPU 동기화 배치 정규화                     |

# 트러블 슈팅

## PATH 관련 warning

```
WARNING: The script {{ SCRIPT_NAME }} is installed in '/home/{{ USER }}/.local/bin' which is not on PATH.
Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
```

단순히 path를 못 찾는다는 말이므로 `export PATH=$PATH:/home/{{ USER }}/.local/bin`를 해준다.

bash에 그대로 써도 문제 없지만, 세션을 다시 시작하면 다시 원점이니 `.bashrc` 등에 넣어주자.
