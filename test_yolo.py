from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

print("=== YOLO 모델 테스트 ===\n")

# 1. YOLO 모델 로드
model_path = r"C:\SmartNougat\new_nougat-main\models\.cache\yolo_v8_formula_det_ft.pt"
print(f"1. 모델 경로: {model_path}")

try:
    model = YOLO(model_path)
    print("[OK] YOLO 모델 로드 성공")
    print(f"모델 정보: {model.model.model[-1].nc} classes")
except Exception as e:
    print(f"[FAIL] 모델 로드 실패: {e}")
    exit(1)

# 2. 테스트 이미지 준비
test_images = [
    r"C:\Nougat_result\1_AI_smartnougat_20250718_183201\pages\page_0.png",
    r"C:\Nougat_result\1_AI_smartnougat_20250718_130808\pages\page_0.png",
    r"C:\Nougat_result\1_AI_smartnougat_20250718_125617\pages\page_0.png"
]

print("\n2. 테스트 이미지:")
for img_path in test_images:
    if os.path.exists(img_path):
        print(f"[OK] {img_path}")
    else:
        print(f"[FAIL] {img_path} - 파일 없음")

# 3. 각 이미지에 대해 예측 수행
print("\n3. YOLO 예측 테스트:")

for img_path in test_images:
    if not os.path.exists(img_path):
        continue
        
    print(f"\n이미지: {os.path.basename(img_path)}")
    
    # 이미지 로드
    img = Image.open(img_path)
    img_array = np.array(img)
    print(f"  이미지 크기: {img_array.shape}")
    
    # 다양한 신뢰도 임계값으로 테스트
    for conf_threshold in [0.1, 0.25, 0.5]:
        results = model.predict(
            img_array,
            imgsz=1888,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # 결과 분석
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    detections.append({
                        'bbox': box.xyxy[0].tolist(),
                        'conf': float(box.conf),
                        'class': int(box.cls)
                    })
        
        print(f"  신뢰도 {conf_threshold}: {len(detections)}개 감지")
        
        # 감지된 객체 정보 출력
        for i, det in enumerate(detections[:3]):  # 최대 3개만 출력
            print(f"    - 객체 {i+1}: conf={det['conf']:.3f}, bbox={[int(x) for x in det['bbox']]}")

# 4. 모델 클래스 정보
print("\n4. 모델 클래스 정보:")
try:
    if hasattr(model.model, 'names'):
        print(f"클래스 이름: {model.model.names}")
    else:
        print("클래스 이름 정보 없음")
except Exception as e:
    print(f"클래스 정보 오류: {e}")

# 5. 낮은 신뢰도로 전체 스캔
print("\n5. 매우 낮은 신뢰도(0.01)로 스캔:")
if os.path.exists(test_images[0]):
    img = Image.open(test_images[0])
    img_array = np.array(img)
    
    results = model.predict(
        img_array,
        imgsz=1888,
        conf=0.01,
        save=False,
        verbose=False
    )
    
    all_detections = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                all_detections.append(float(box.conf))
    
    if all_detections:
        print(f"총 {len(all_detections)}개 감지")
        print(f"신뢰도 범위: {min(all_detections):.4f} ~ {max(all_detections):.4f}")
    else:
        print("아무것도 감지되지 않음")

print("\n=== 테스트 완료 ===")