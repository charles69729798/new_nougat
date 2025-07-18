import requests
import os
import sys

def download_file_with_progress(url, dest_path):
    """대용량 파일을 진행률과 함께 다운로드"""
    try:
        # 파일 크기 확인
        response = requests.head(url, allow_redirects=True)
        file_size = int(response.headers.get('content-length', 0))
        
        print(f"Downloading YOLO model...")
        print(f"URL: {url}")
        print(f"Destination: {dest_path}")
        print(f"Expected size: {file_size / 1024 / 1024:.1f} MB")
        
        # 스트리밍으로 다운로드
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        downloaded = 0
        chunk_size = 8192
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 진행률 표시
                    if file_size > 0:
                        percent = (downloaded / file_size) * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = file_size / 1024 / 1024
                        sys.stdout.write(f'\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
                        sys.stdout.flush()
        
        print("\n\nDownload complete!")
        actual_size = os.path.getsize(dest_path)
        print(f"File size: {actual_size / 1024 / 1024:.1f} MB")
        
        if actual_size < 1000000:  # 1MB 미만이면 오류
            print("ERROR: File size is too small. Download may have failed.")
            return False
            
        return True
        
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

if __name__ == "__main__":
    # YOLO 모델 URL
    url = "https://github.com/opendatalab/PDF-Extract-Kit/releases/download/PDFExtractv1.0/yolo_v8_formula_det_ft.pt"
    
    # 저장 경로
    models_dir = r"C:\SmartNougat\new_nougat-main\models"
    dest_path = os.path.join(models_dir, "yolo_v8_formula_det_ft.pt")
    
    # 디렉토리 확인
    if not os.path.exists(models_dir):
        print(f"Creating directory: {models_dir}")
        os.makedirs(models_dir)
    
    # 기존 파일 확인
    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)
        if existing_size > 1000000:  # 1MB 이상이면 이미 있는 것으로 간주
            print(f"Model already exists: {dest_path}")
            print(f"Size: {existing_size / 1024 / 1024:.1f} MB")
            print("If you want to re-download, please delete the existing file first.")
            sys.exit(0)
        else:
            print(f"Existing file is too small ({existing_size} bytes). Re-downloading...")
            os.remove(dest_path)
    
    # 다운로드 실행
    success = download_file_with_progress(url, dest_path)
    
    if success:
        print("\nYOLO model downloaded successfully!")
        print("You can now run SmartNougat GUI.")
    else:
        print("\nDownload failed. Please try again or download manually.")
        print(f"Manual download URL: {url}")
        print(f"Save to: {dest_path}")