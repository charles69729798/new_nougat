import os
import sys

print("=== PDF-Extract-Kit 수식 감지 모델 다운로드 ===\n")

try:
    from huggingface_hub import snapshot_download
    print("[OK] huggingface_hub 모듈 설치됨")
except ImportError:
    print("[FAIL] huggingface_hub가 설치되지 않았습니다.")
    print("설치 명령: pip install huggingface_hub")
    sys.exit(1)

# 저장 경로
download_dir = r"C:\SmartNougat\new_nougat-main\pdf-extract-kit-models"
print(f"다운로드 경로: {download_dir}")

try:
    print("\nHugging Face에서 수식 감지 모델 다운로드 중...")
    print("(처음 다운로드 시 시간이 걸릴 수 있습니다)")
    
    # MFD/YOLO 모델만 다운로드
    snapshot_download(
        repo_id='opendatalab/pdf-extract-kit-1.0',
        local_dir=download_dir,
        allow_patterns='models/MFD/YOLO/*',
        max_workers=5
    )
    
    print("\n다운로드 완료!")
    
    # 다운로드된 파일 확인
    yolo_dir = os.path.join(download_dir, "models", "MFD", "YOLO")
    if os.path.exists(yolo_dir):
        files = os.listdir(yolo_dir)
        print(f"\n다운로드된 파일들:")
        for f in files:
            size = os.path.getsize(os.path.join(yolo_dir, f)) / 1024 / 1024
            print(f"  - {f}: {size:.1f} MB")
        
        # weights.pt를 올바른 위치로 복사
        weights_path = os.path.join(yolo_dir, "weights.pt")
        if os.path.exists(weights_path):
            import shutil
            
            # 모델 디렉토리에 복사
            dest1 = r"C:\SmartNougat\new_nougat-main\models\yolo_v8_formula_det_ft.pt"
            shutil.copy2(weights_path, dest1)
            print(f"\n[OK] 모델 복사됨: {dest1}")
            
            # 캐시 디렉토리에도 복사 (기존 파일 덮어쓰기)
            cache_dir = r"C:\SmartNougat\new_nougat-main\models\.cache"
            os.makedirs(cache_dir, exist_ok=True)
            dest2 = os.path.join(cache_dir, "yolo_v8_formula_det_ft.pt")
            shutil.copy2(weights_path, dest2)
            print(f"[OK] 캐시에도 복사됨: {dest2}")
            
            print("\n[성공] 진짜 수식 감지 모델 설치 완료!")
            print("이제 SmartNougat이 수식을 제대로 감지할 수 있습니다.")
        else:
            print("\n[경고] weights.pt 파일을 찾을 수 없습니다.")
            print(f"다운로드된 파일들을 확인하세요: {yolo_dir}")
    else:
        print(f"\n[오류] YOLO 디렉토리를 찾을 수 없습니다: {yolo_dir}")
        
except Exception as e:
    print(f"\n[오류] 다운로드 실패: {e}")
    print("\n대안: 브라우저에서 직접 다운로드하세요:")
    print("1. https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/tree/main/models/MFD/YOLO")
    print("2. weights.pt 파일 다운로드")
    print("3. C:\\SmartNougat\\new_nougat-main\\models\\yolo_v8_formula_det_ft.pt로 저장")