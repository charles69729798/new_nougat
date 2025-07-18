#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nougat 모델 다운로드 스크립트
"""
from huggingface_hub import snapshot_download
import os
from pathlib import Path
import shutil

def download_nougat_model():
    """Norm/nougat-latex-base 모델 다운로드"""
    
    print("=== Nougat LaTeX 모델 다운로드 ===\n")
    
    model_name = "Norm/nougat-latex-base"
    
    # 다운로드 경로
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    
    try:
        print(f"모델 다운로드 중: {model_name}")
        print("(처음 다운로드 시 시간이 걸릴 수 있습니다)")
        
        # Hugging Face Hub에서 다운로드
        downloaded_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True
        )
        
        print(f"\n다운로드 완료: {downloaded_path}")
        
        # 다운로드된 파일 확인
        required_files = [
            'pytorch_model.bin',
            'config.json',
            'tokenizer_config.json',
            'tokenizer.json',
            'preprocessor_config.json'
        ]
        
        print("\n다운로드된 파일 확인:")
        for file in required_files:
            file_path = Path(downloaded_path) / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"✓ {file} ({size_mb:.1f} MB)")
            else:
                print(f"✗ {file} (없음)")
        
        # models 폴더로 복사 옵션
        models_dir = Path("C:/SmartNougat/new_nougat-main/models_norm")
        print(f"\n로컬 models 폴더로 복사하시겠습니까? {models_dir}")
        print("(기존 models 폴더는 models_backup으로 백업됩니다)")
        
        return downloaded_path
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        return None

if __name__ == "__main__":
    result = download_nougat_model()
    if result:
        print(f"\n성공! 모델 위치: {result}")
        print("\n이제 SmartNougat이 정상 작동할 것입니다.")
    else:
        print("\n실패! 인터넷 연결을 확인하세요.")