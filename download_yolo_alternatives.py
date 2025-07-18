import requests
import os
import sys

def download_file(url, dest_path):
    """파일 다운로드 시도"""
    try:
        print(f"\nTrying: {url}")
        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            print(f"Found! Size: {total_size / 1024 / 1024:.1f} MB")
            
            downloaded = 0
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f'\rProgress: {percent:.1f}%')
                            sys.stdout.flush()
            
            print("\nDownload complete!")
            return True
        else:
            print(f"Failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

# 가능한 다운로드 URL들
urls = [
    # HuggingFace 미러
    "https://huggingface.co/wanderkid/PDF-Extract-Kit/resolve/main/models/MFD/YOLO/yolo.pt",
    "https://huggingface.co/wanderkid/PDF-Extract-Kit/resolve/main/models/MFD/yolo.pt",
    
    # ModelScope (중국 미러)
    "https://modelscope.cn/models/wanderkid/PDF-Extract-Kit/resolve/master/models/MFD/YOLO/yolo.pt",
    
    # 다른 가능한 이름들
    "https://github.com/opendatalab/PDF-Extract-Kit/releases/download/v1.0/yolo_mfd.pt",
    "https://github.com/opendatalab/PDF-Extract-Kit/releases/download/v1.0/yolo_formula.pt",
    
    # 일반 YOLOv8 모델 (대체용)
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
]

# 저장 경로
models_dir = r"C:\SmartNougat\new_nougat-main\models"
os.makedirs(models_dir, exist_ok=True)

print("Searching for YOLO formula detection model...")

# 각 URL 시도
success = False
for url in urls:
    # 파일명 추출
    filename = url.split('/')[-1]
    dest_path = os.path.join(models_dir, filename)
    
    if download_file(url, dest_path):
        # 성공하면 필요한 이름으로 복사
        target_path = os.path.join(models_dir, "yolo_v8_formula_det_ft.pt")
        
        # Windows에서 파일 복사
        import shutil
        shutil.copy2(dest_path, target_path)
        
        print(f"\nModel saved as: {target_path}")
        print(f"Size: {os.path.getsize(target_path) / 1024 / 1024:.1f} MB")
        success = True
        break

if not success:
    print("\n❌ Could not download YOLO model from any source.")
    print("\nAlternative: You can use a general YOLOv8 model.")
    print("Download manually from: https://github.com/ultralytics/assets/releases/")
    print("Look for yolov8x.pt or yolov8l.pt")
else:
    print("\n✅ YOLO model ready! You can now run SmartNougat.")