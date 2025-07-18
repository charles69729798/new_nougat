from transformers import VisionEncoderDecoderModel, AutoTokenizer
import sys
import os

# nougat_latex 경로 추가
sys.path.insert(0, r"C:\SmartNougat\new_nougat-main\src")

try:
    from nougat_latex import NougatLaTexProcessor
except:
    NougatLaTexProcessor = None

print("=== Norm/nougat-latex-base 모델 다운로드 ===\n")

model_name = "Norm/nougat-latex-base"
save_dir = r"C:\SmartNougat\new_nougat-main\models_norm"

try:
    print(f"1. 모델 다운로드 중: {model_name}")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    print(f"2. 토크나이저 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"3. Processor 다운로드 중...")
    try:
        processor = NougatLaTexProcessor.from_pretrained(model_name)
    except:
        processor = None
        print("   (Processor는 선택사항)")
    
    # 로컬에 저장
    print(f"\n4. 로컬 저장: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print("\n[성공] 모델 다운로드 및 저장 완료!")
    print(f"저장 위치: {save_dir}")
    
    # models 폴더로 복사
    import shutil
    models_dir = r"C:\SmartNougat\new_nougat-main\models"
    
    print(f"\n5. 기존 models 폴더 백업...")
    if os.path.exists(models_dir):
        backup_dir = models_dir + "_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(models_dir, backup_dir)
        print(f"   백업됨: {backup_dir}")
    
    print(f"6. 새 모델 복사...")
    shutil.copytree(save_dir, models_dir)
    print(f"   복사됨: {models_dir}")
    
    print("\n[완료] 이제 SmartNougat이 정상 작동할 것입니다!")
    
except Exception as e:
    print(f"\n[오류] 다운로드 실패: {e}")
    import traceback
    traceback.print_exc()