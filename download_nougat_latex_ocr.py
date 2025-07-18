import os
import sys
import zipfile
import shutil
import requests

print("=== nougat-latex-ocr 모듈 다운로드 ===\n")

# GitHub에서 ZIP 다운로드
url = "https://github.com/NormXU/nougat-latex-ocr/archive/refs/heads/main.zip"
zip_path = "nougat-latex-ocr.zip"

print("1. GitHub에서 다운로드 중...")
try:
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    downloaded = 0
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f'\r  진행률: {percent:.1f}%')
                    sys.stdout.flush()
    
    print("\n[OK] 다운로드 완료")
except Exception as e:
    print(f"[FAIL] 다운로드 실패: {e}")
    sys.exit(1)

# ZIP 압축 해제
print("\n2. 압축 해제 중...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("[OK] 압축 해제 완료")
except Exception as e:
    print(f"[FAIL] 압축 해제 실패: {e}")
    sys.exit(1)

# nougat_latex 모듈을 src 디렉토리로 복사
print("\n3. 모듈 설치 중...")
try:
    source_dir = os.path.join("nougat-latex-ocr-main", "nougat_latex")
    dest_dir = os.path.join("src", "nougat_latex")
    
    # 기존 디렉토리 삭제
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    
    # 복사
    shutil.copytree(source_dir, dest_dir)
    print(f"[OK] nougat_latex 모듈 복사됨: {dest_dir}")
    
    # 정상 작동하는 환경의 올바른 Nougat 모델 경로
    print("\n4. Nougat 모델 정보:")
    print("- 모델명: Norm/nougat-latex-base")
    print("- Hugging Face에서 다운로드 필요")
    
except Exception as e:
    print(f"[FAIL] 모듈 설치 실패: {e}")
    sys.exit(1)

# 정리
try:
    os.remove(zip_path)
    shutil.rmtree("nougat-latex-ocr-main")
    print("\n[OK] 임시 파일 정리 완료")
except:
    pass

print("\n=== 설치 완료 ===")
print("\n중요: transformers의 ChannelDimension 문제는 다음과 같이 해결할 수 있습니다:")
print("1. transformers 버전이 4.34.0인지 확인")
print("2. 이미지가 RGB 모드인지 확인")
print("3. processor 대신 직접 전처리하는 방법 사용")