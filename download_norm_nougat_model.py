from huggingface_hub import snapshot_download
import os
import shutil

print("=== Norm/nougat-latex-base 모델 다운로드 ===\n")

# 다운로드 경로
cache_dir = r"C:\SmartNougat\new_nougat-main\models_cache"
target_dir = r"C:\SmartNougat\new_nougat-main\models"

try:
    print("Hugging Face에서 Norm/nougat-latex-base 모델 다운로드 중...")
    print("(처음 다운로드 시 시간이 걸릴 수 있습니다)\n")
    
    # Norm/nougat-latex-base 모델 다운로드
    downloaded_path = snapshot_download(
        repo_id='Norm/nougat-latex-base',
        cache_dir=cache_dir,
        max_workers=5
    )
    
    print(f"\n다운로드 완료: {downloaded_path}")
    
    # 필요한 파일들을 models 디렉토리로 복사
    required_files = [
        'config.json',
        'generation_config.json',
        'preprocessor_config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'pytorch_model.bin'
    ]
    
    print(f"\n파일 복사 중: {downloaded_path} -> {target_dir}")
    
    for file in required_files:
        src = os.path.join(downloaded_path, file)
        dst = os.path.join(target_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[OK] {file} 복사됨")
        else:
            print(f"[경고] {file} 없음")
    
    # safetensors 파일 확인
    safetensors_file = os.path.join(downloaded_path, 'model.safetensors')
    if os.path.exists(safetensors_file):
        shutil.copy2(safetensors_file, os.path.join(target_dir, 'model.safetensors'))
        print("[OK] model.safetensors 복사됨")
    
    print("\n[성공] Norm/nougat-latex-base 모델 설치 완료!")
    print("이제 SmartNougat이 올바른 Nougat 모델을 사용할 수 있습니다.")
    
except Exception as e:
    print(f"\n[오류] 다운로드 실패: {e}")
    print("\n대안:")
    print("1. https://huggingface.co/Norm/nougat-latex-base 방문")
    print("2. Files and versions 탭에서 파일 다운로드")
    print(f"3. {target_dir} 디렉토리에 저장")