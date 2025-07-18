import sys
import os
sys.path.insert(0, r"C:\SmartNougat\new_nougat-main\src")
os.environ['PYTHONPATH'] = r"C:\SmartNougat\new_nougat-main\nougat-latex-ocr;C:\SmartNougat\new_nougat-main\src"

from smartnougat_0715 import SmartNougatStandalone

# PDF 파일 직접 테스트
pdf_path = r"C:\test\1_AI.pdf"
if not os.path.exists(pdf_path):
    print(f"PDF 파일이 없습니다: {pdf_path}")
    print("먼저 DOCX를 PDF로 변환해주세요.")
else:
    print(f"PDF 파일로 직접 테스트: {pdf_path}")
    
    # SmartNougat 초기화
    processor = SmartNougatStandalone(device="cpu")
    
    # 페이지 4 처리
    result = processor.process_document(pdf_path, pages="4")
    
    print(f"\n결과 저장 위치: {result}")