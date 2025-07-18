from transformers import VisionEncoderDecoderModel, AutoTokenizer
import os

print("=== Nougat 모델 로드 테스트 ===\n")

# 1. 로컬 모델 테스트
local_path = r"C:\SmartNougat\new_nougat-main\models"
print(f"1. 로컬 경로 테스트: {local_path}")

if os.path.exists(local_path):
    files = os.listdir(local_path)
    print(f"   파일들: {[f for f in files if f.endswith(('.json', '.bin', '.safetensors'))]}")
    
    try:
        model = VisionEncoderDecoderModel.from_pretrained(local_path)
        print("   [OK] 모델 로드 성공")
        print(f"   모델 타입: {type(model)}")
    except Exception as e:
        print(f"   [FAIL] 모델 로드 실패: {e}")

# 2. Norm/nougat-latex-base 테스트
print("\n2. Norm/nougat-latex-base 모델 테스트:")
try:
    # 캐시에서 먼저 시도
    model = VisionEncoderDecoderModel.from_pretrained(
        "Norm/nougat-latex-base",
        local_files_only=True
    )
    print("   [OK] 캐시에서 로드 성공")
except Exception as e:
    print(f"   [INFO] 캐시에 없음: {e}")
    print("   온라인에서 다운로드 필요")

# 3. 모델 경로 확인
print("\n3. Hugging Face 캐시 경로:")
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    models = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
    for m in models:
        if "nougat" in m.lower():
            print(f"   - {m}")

print("\n=== 테스트 완료 ===")