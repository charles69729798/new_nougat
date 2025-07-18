import win32com.client
import os

# Word COM 객체 테스트
try:
    print("1. Word.Application 생성 시도...")
    word = win32com.client.gencache.EnsureDispatch("Word.Application")
    print("   성공! Word 애플리케이션 생성됨")
    
    # Word 버전 확인
    print(f"2. Word 버전: {word.Version}")
    
    # Word 표시 설정
    word.Visible = True  # 디버깅을 위해 표시
    word.DisplayAlerts = 0  # wdAlertsNone
    
    print("3. 문서 열기 시도...")
    docx_path = r"C:\test\1_AI.docx"
    
    if not os.path.exists(docx_path):
        print(f"   오류: 파일이 없습니다 - {docx_path}")
    else:
        # 문서 열기
        doc = word.Documents.Open(docx_path)
        print("   성공! 문서가 열렸습니다")
        
        # 문서 정보
        print(f"4. 문서 이름: {doc.Name}")
        print(f"   페이지 수: {doc.ComputeStatistics(2)}")  # 2 = wdStatisticPages
        
        # PDF로 저장 시도
        pdf_path = r"C:\test\1_AI.pdf"
        print(f"5. PDF로 저장 시도: {pdf_path}")
        
        try:
            # SaveAs2 메서드 사용 (Word 2010+)
            doc.SaveAs2(pdf_path, FileFormat=17)  # 17 = wdFormatPDF
            print("   성공! PDF로 저장됨")
        except:
            print("   SaveAs2 실패, SaveAs 시도...")
            try:
                doc.SaveAs(pdf_path, 17)
                print("   성공! PDF로 저장됨")
            except Exception as e:
                print(f"   SaveAs도 실패: {e}")
        
        # 문서 닫기
        doc.Close(0)  # 0 = wdDoNotSaveChanges
        print("6. 문서 닫기 완료")
        
    # Word 종료
    word.Quit()
    print("7. Word 종료 완료")
    
except Exception as e:
    print(f"\n오류 발생!")
    print(f"타입: {type(e).__name__}")
    print(f"메시지: {str(e)}")
    
    # 더 자세한 정보
    import traceback
    print("\n상세 오류:")
    traceback.print_exc()
    
    # COM 오류인 경우
    if hasattr(e, 'excepinfo'):
        print(f"\nCOM 오류 정보: {e.excepinfo}")