import win32com.client
import os

# Word 애플리케이션 시작
word = win32com.client.Dispatch("Word.Application")
word.Visible = False

try:
    # DOCX 파일 열기
    docx_path = r"C:\test\1_AI.docx"
    pdf_path = r"C:\test\1_AI.pdf"
    
    print(f"변환 중: {docx_path} -> {pdf_path}")
    
    # 절대 경로로 변환
    docx_path = os.path.abspath(docx_path)
    pdf_path = os.path.abspath(pdf_path)
    
    doc = word.Documents.Open(docx_path)
    
    # PDF로 내보내기
    doc.ExportAsFixedFormat(pdf_path, ExportFormat=17)  # 17 = wdExportFormatPDF
    doc.Close()
    
    print(f"변환 완료: {pdf_path}")
    
except Exception as e:
    print(f"오류 발생: {e}")
    
finally:
    word.Quit()