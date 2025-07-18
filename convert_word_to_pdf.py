import win32com.client
import os
import pythoncom

# COM 초기화
pythoncom.CoInitialize()

try:
    # Word 애플리케이션 시작
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    
    # DOCX 파일 경로
    docx_path = r"C:\test\1_AI.docx"
    pdf_path = r"C:\test\1_AI.pdf"
    
    # 절대 경로로 변환
    docx_path = os.path.abspath(docx_path)
    pdf_path = os.path.abspath(pdf_path)
    
    print(f"입력 파일: {docx_path}")
    print(f"출력 파일: {pdf_path}")
    
    # 파일 존재 확인
    if not os.path.exists(docx_path):
        print(f"오류: 입력 파일이 없습니다 - {docx_path}")
    else:
        # 문서 열기
        doc = word.Documents.Open(docx_path)
        
        # PDF로 저장 (ExportAsFixedFormat 사용)
        # 0 = wdExportFormatPDF
        doc.ExportAsFixedFormat(OutputFileName=pdf_path, 
                               ExportFormat=0,
                               OpenAfterExport=False,
                               OptimizeFor=0,
                               BitmapMissingFonts=True)
        
        # 문서 닫기
        doc.Close(SaveChanges=False)
        
        print(f"변환 완료: {pdf_path}")
        print(f"파일 크기: {os.path.getsize(pdf_path) / 1024:.1f} KB")
        
except Exception as e:
    print(f"오류 발생: {type(e).__name__}")
    print(f"오류 메시지: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    try:
        # Word 종료
        word.Quit()
    except:
        pass
    
    # COM 정리
    pythoncom.CoUninitialize()