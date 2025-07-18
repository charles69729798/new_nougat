import sys
import os

print("=== SmartNougat 의존성 테스트 ===\n")

# 1. win32com 테스트
print("1. win32com + MS Word 테스트:")
try:
    import win32com.client
    import pythoncom
    print("[OK] win32com 모듈 import 성공")
    
    # Word 애플리케이션 테스트
    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    print(f"[OK] Word 애플리케이션 실행 성공 (버전: {word.Version})")
    word.Quit()
    pythoncom.CoUninitialize()
except Exception as e:
    print(f"[FAIL] win32com/Word 오류: {e}")

# 2. PyMuPDF 테스트
print("\n2. PyMuPDF 테스트:")
try:
    import fitz
    print(f"[OK] PyMuPDF import 성공 (버전: {fitz.version[0]})")
except Exception as e:
    print(f"[FAIL] PyMuPDF 오류: {e}")

# 3. YOLO 테스트
print("\n3. YOLO (Ultralytics) 테스트:")
try:
    from ultralytics import YOLO
    print("[OK] YOLO import 성공")
    
    # 모델 파일 확인
    model_path = r"C:\SmartNougat\new_nougat-main\models\.cache\yolo_v8_formula_det_ft.pt"
    if os.path.exists(model_path):
        print(f"[OK] YOLO 모델 파일 존재: {model_path}")
    else:
        print(f"[FAIL] YOLO 모델 파일 없음: {model_path}")
except Exception as e:
    print(f"[FAIL] YOLO 오류: {e}")

# 4. Transformers/Nougat 테스트
print("\n4. Transformers/Nougat 테스트:")
try:
    from transformers import VisionEncoderDecoderModel, AutoTokenizer
    print("[OK] Transformers import 성공")
    
    # 로컬 모델 경로 확인
    model_path = r"C:\SmartNougat\new_nougat-main\models"
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"[OK] Nougat 모델 파일 존재: {model_path}")
    else:
        print(f"[FAIL] Nougat 모델 파일 없음: {model_path}")
except Exception as e:
    print(f"[FAIL] Transformers 오류: {e}")

# 5. 기타 의존성
print("\n5. 기타 필수 의존성:")
modules = ["numpy", "PIL", "torch", "docx2pdf"]
for module in modules:
    try:
        __import__(module)
        print(f"[OK] {module} import 성공")
    except Exception as e:
        print(f"[FAIL] {module} 오류: {e}")

# 6. Python 경로 확인
print("\n6. Python 경로:")
print(f"Python 버전: {sys.version}")
print(f"현재 작업 디렉토리: {os.getcwd()}")

# 7. nougat_latex 모듈 확인
print("\n7. nougat_latex 모듈:")
sys.path.append(r"C:\SmartNougat\new_nougat-main\src")
try:
    import nougat_latex
    print("[OK] nougat_latex import 성공")
    from nougat_latex.util import process_raw_latex_code
    print("[OK] process_raw_latex_code import 성공")
except Exception as e:
    print(f"[FAIL] nougat_latex 오류: {e}")

print("\n=== 테스트 완료 ===")