import win32com.client
import pythoncom
import time
import os

def convert_docx_to_pdf(docx_path, pdf_path):
    """DOCX를 PDF로 변환"""
    pythoncom.CoInitialize()
    
    try:
        print("Word 애플리케이션 시작...")
        # DispatchEx를 사용하여 새 인스턴스 생성
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = False
        
        print(f"문서 열기: {docx_path}")
        # ReadOnly로 열기
        doc = word.Documents.Open(docx_path, ReadOnly=True)
        
        # 잠시 대기
        time.sleep(1)
        
        print(f"PDF로 저장: {pdf_path}")
        # SaveAs2 대신 ExportAsFixedFormat2 사용
        doc.SaveAs2(FileName=pdf_path, FileFormat=17)  # 17 = wdFormatPDF
        
        print("문서 닫기...")
        doc.Close(SaveChanges=False)
        
        print("Word 종료...")
        word.Quit()
        
        print("변환 완료!")
        return True
        
    except Exception as e:
        print(f"오류: {e}")
        try:
            if 'doc' in locals():
                doc.Close(SaveChanges=False)
            if 'word' in locals():
                word.Quit()
        except:
            pass
        return False
    finally:
        pythoncom.CoUninitialize()

# 실행
if __name__ == "__main__":
    docx_file = r"C:\test\1_AI.docx"
    pdf_file = r"C:\test\1_AI.pdf"
    
    if os.path.exists(docx_file):
        success = convert_docx_to_pdf(docx_file, pdf_file)
        if success and os.path.exists(pdf_file):
            print(f"\nPDF 파일 생성됨: {pdf_file}")
            print(f"파일 크기: {os.path.getsize(pdf_file) / 1024:.1f} KB")
    else:
        print(f"입력 파일이 없습니다: {docx_file}")