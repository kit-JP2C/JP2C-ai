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

## 사용 데이터 셋

> 오픈소스를 전문 용어로 긴빠이라고 한다 아쎄이

[Mahjong Grayscale Filtered - Roboflow](https://app.roboflow.com/mahjong-9gpqq/yolo_mahjong_grayscale-7ahri/4)

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

# flutter에 적용

## 필수 파일 목록

1. 이 프로젝트의 labels.txt
2. [변환](#변환) 후 나온 tflite 파일

## 공통

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/model.tflite
    - assets/labels.txt
```

## flutter에서 파일 읽기

> labels.txt와 \*\*\*.tflite 파일은 assets에 존재해야 합니다.

### labels.txt

```dart
import 'package:flutter/services.dart' show rootBundle;

Future<List<String>> loadLabels() async {
  final labelsData = await rootBundle.loadString('assets/labels.txt');
  return labelsData.split('\n'); // 줄 단위로 분리
}
```

### \*\*\*.tflite

> tflite 파일은 binary 파일이므로 load 후 uint list로 변경합니다.

> tflite는 [pub.dev - tflite_flutter](https://pub.dev/packages/tflite_flutter/install)를 설치해주세요.

```dart
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data';

Future<Uint8List> loadModel() async {
  final modelData = await rootBundle.load('assets/model.tflite');
  return modelData.buffer.asUint8List();
}
```

## 이걸 싱글톤 패턴으로 관리하자

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;

class YOLOModel {
  static final YOLOModel _instance = YOLOModel._internal();
  static YOLOModel get instance => _instance;

  // flutter에서 color는 0xAARRGGBB 형태로 표현됨 -> 0xfff 같은걸로는 안된다
  static const _BLACK = 0xFF00000000;

  Interpreter? interpreter; // tflite 인스턴스
  List<String>? labels;     // 라벨 목록

  YOLOModel._internal();

  // 초기화: 모델 + labels
  Future<void> loadModelAndLabels() async {
    if (interpreter != null && labels != null) return;

    interpreter = await Interpreter.fromAsset('assets/your_model.tflite');

    final String raw = await rootBundle.loadString('assets/labels.txt');
    labels = raw.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
  }

  // 추론 실행
  Future<List<String>> predict(ui.Image image, {double threshold = 0.5}) async {
    if (interpreter == null) throw Exception('Interpreter not loaded');
    if (labels == null) throw Exception('Labels not loaded');

    // 1️⃣ 이미지 전처리 (YOLO 입력 크기, float32 등)
    final input = _imageToByteListFloat32(image, 640, 640); // 예: 640x640

    // 2️⃣ 출력 배열 생성 (YOLO 모델 구조에 맞춰)
    // 현재 YOLO 모델의 구조는 TensorSpec(shape=(1, 39, 8400), dtype=tf.float32, name=None)
    final output = List.filled(1 * 39 * 8400, 0.0).reshape([1, 39, 8400]);

    // 3️⃣ 추론 실행
    interpreter!.run(input, output);

    // 4️⃣ 후처리: threshold로 필터링 + class index → label
    List<String> detected = [];
    for (var box in output[0]) { // 첫 번째 이미지 추론 결과
      // box: [x, y, w, h, score, classNo]
      double score = box[4];
      int classIndex = box[5].toInt();
      if (score > threshold) {
        detected.add('${labels![classIndex]} (${(score*100).toStringAsFixed(1)}%)');
      }
    }

    return detected;
  }

  Future<Float32List> _imageToByteListFloat32(ui.Image image, int inputSize) async {
    final int originalWidth = image.width;
    final int originalHeight = image.height;

    double scale = inputSize / (originalWidth > originalHeight ? originalWidth : originalHeight);
    int resizedWidth = (originalWidth * scale).round();
    int resizedHeight = (originalHeight * scale).round();

    int padLeft = (inputSize - resizedWidth) ~/ 2;
    int padTop = (inputSize - resizedHeight) ~/ 2;

    final ui.PictureRecorder recoder = ui.PictureRecorder();
    final ui.Canvas canvas = ui.Canvas(recorder);
    final ui.Paint paint = ui.Paint();

    // 일단 모든 영역을 검은색으로 채워
    canvas.drawRect(
      // LTWH = (Left, Top, Width, Height)
      ui.Rect.fromLTWH(0, 0, inputSize.toDouble(), inputSize.toDouble()),
      paint..color = ui.Color(_BLACK)
    );
    // 그리고 이미지를 덮어 씌워
    canvas.drawImageRect(
      image, // source
      ui.Rect.fromLTWH(0, 0, originalWidth.toDouble(), originalHeight.toDouble()), // source에서 가져올 영역
      ui.Rect.fromLTWH(padLeft.toDouble(), padTop.toDouble(), resizedWidth.toDouble(), resizedHeight.toDouble()), // 캔버스에 붙여넣을 영역
      paint, // 적용
    );

    // 결과 이미지 가져오기
    final ui.Image resizedImage = await recorder.endRecording().toImage(inputSize, inputSize);
    final ByteData? byteData = await resizedImage.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) throw Exception("Failed to convert image to byte data.");

    final Uint8List pixels = byteData.buffer.asUint8List();
    final Float32List temp = Float32List(1 * inputSize * inputSize * 3);
    Float32List buffer Float32List(1 * inputSize * inputSize * 3);
    // 우리는 rgba 에서 rgb만 필요하다.
    int pixelIndex = 0;
    for (int i = 0; i < pixels.length; i += 4) {
      temp[pixelIndex++] = pixels[i] / 255.0;     // R channel
      temp[pixelIndex++] = pixels[i + 1] / 255.0; // G channel
      temp[pixelIndex++] = pixels[i + 2] / 255.0; // B channel
    }
    buffer = temp;

    // tflite 입력값에 맞게 변환. 만약 결과가 이상하다면 주석처리 후 재시도 - BHCW
    pixelIndex = 0;
    for (int channel = 0; channel < 3; channel++) {
      for (int i = 0; i < temp.length / 3; i++) {
        buffer[pixelIndex++] = temp[i*3 + channel];
      }
    }
    return buffer;
  }

  void close() {
    interpreter?.close();
    interpreter = null;
    labels = null;
  }
}
```

# 트러블 슈팅

## PATH 관련 warning

```
WARNING: The script {{ SCRIPT_NAME }} is installed in '/home/{{ USER }}/.local/bin' which is not on PATH.
Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
```

단순히 path를 못 찾는다는 말이므로 `export PATH=$PATH:/home/{{ USER }}/.local/bin`를 해준다.

bash에 그대로 써도 문제 없지만, 세션을 다시 시작하면 다시 원점이니 `.bashrc` 등에 넣어주자.
