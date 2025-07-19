#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartNougat GUI Batch - 폴더 일괄 처리 버전
- 폴더 선택 후 내부 DOCX/PDF 파일 모두 순차 처리
- 기존 GUI와 동일한 레이아웃
- 전체 진행률 및 현재 파일 진행률 표시
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import re
import time
import queue
from pathlib import Path
import json
import base64
from datetime import datetime
from bs4 import BeautifulSoup
import glob

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

class SmartNougatBatchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartNougat GUI Batch - 폴더 일괄 처리")
        self.root.geometry("900x750")
        
        # 설정 로드
        self.config = self.load_config()
        
        # 변수들
        self.selected_folder = tk.StringVar()
        self.page_range = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.is_processing = False
        self.process = None
        self.output_queue = queue.Queue()
        self.start_time = None
        
        # 통계
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_formulas': 0,
            'processed_formulas': 0,
            'current_file': '',
            'files_list': []
        }
        
        self.setup_ui()
        self.start_output_reader()
        
    def load_config(self):
        """설정 파일 로드"""
        config_file = Path(__file__).parent / "config.json"
        default_config = {
            "default_output_folder": "",
            "default_page_range": "",
            "window_geometry": "900x750"
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return default_config
        return default_config
        
    def save_config(self):
        """설정 파일 저장"""
        config_file = Path(__file__).parent / "config.json"
        self.config["default_output_folder"] = self.output_folder.get()
        self.config["default_page_range"] = self.page_range.get()
        self.config["window_geometry"] = self.root.geometry()
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except:
            pass
            
    def setup_ui(self):
        """사용자 인터페이스 설정"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 제목
        title_label = ttk.Label(main_frame, text="SmartNougat 폴더 일괄 처리", 
                              font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 폴더 선택
        ttk.Label(main_frame, text="처리할 폴더 선택:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        folder_frame.columnconfigure(0, weight=1)
        
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.selected_folder, 
                                     state="readonly", width=60)
        self.folder_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(folder_frame, text="폴더 선택", 
                  command=self.browse_folder).grid(row=0, column=1)
        
        # 페이지 범위
        ttk.Label(main_frame, text="페이지 범위:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        page_frame = ttk.Frame(main_frame)
        page_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.page_entry = ttk.Entry(page_frame, textvariable=self.page_range, width=20)
        self.page_entry.grid(row=0, column=0, sticky=tk.W)
        self.page_range.set(self.config.get("default_page_range", ""))
        
        ttk.Label(page_frame, text="(예: 1, 1-5, 1,3,5-7, 비워두면 전체 페이지)").grid(
            row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 출력 폴더
        ttk.Label(main_frame, text="출력 폴더:").grid(row=3, column=0, sticky=tk.W, pady=5)
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_folder, 
                                     state="readonly", width=60)
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(output_frame, text="폴더 선택", 
                  command=self.browse_output_folder).grid(row=0, column=1)
        
        # 파일 목록
        ttk.Label(main_frame, text="처리할 파일 목록:").grid(row=4, column=0, sticky=tk.W, pady=5)
        
        # 파일 리스트박스와 스크롤바
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        list_frame.columnconfigure(0, weight=1)
        
        self.files_listbox = tk.Listbox(list_frame, height=6)
        self.files_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        files_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.files_listbox.yview)
        files_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.files_listbox.config(yscrollcommand=files_scrollbar.set)
        
        # 진행률 섹션
        progress_frame = ttk.LabelFrame(main_frame, text="진행 상황", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(1, weight=1)
        
        # 전체 진행률
        ttk.Label(progress_frame, text="전체 진행률:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.overall_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.overall_progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # 현재 파일 진행률
        ttk.Label(progress_frame, text="현재 파일:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.current_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.current_progress.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # 통계 정보
        stats_frame = ttk.Frame(progress_frame)
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(3, weight=1)
        
        ttk.Label(stats_frame, text="파일:").grid(row=0, column=0, sticky=tk.W)
        self.files_label = ttk.Label(stats_frame, text="0/0")
        self.files_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        ttk.Label(stats_frame, text="현재 처리 중:").grid(row=0, column=2, sticky=tk.W)
        self.current_file_label = ttk.Label(stats_frame, text="없음")
        self.current_file_label.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        
        # 제어 버튼
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="일괄 처리 시작", 
                                      command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="중지", 
                                     command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="로그 지우기", 
                                      command=self.clear_log)
        self.clear_button.grid(row=0, column=2, padx=(0, 10))
        
        self.open_folder_button = ttk.Button(button_frame, text="결과 폴더 열기", 
                                           command=self.open_results_folder, state="disabled")
        self.open_folder_button.grid(row=0, column=3)
        
        # 로그 영역
        log_frame = ttk.LabelFrame(main_frame, text="처리 로그", padding="5")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=100)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 리사이징을 위한 그리드 가중치 설정
        main_frame.rowconfigure(7, weight=1)
        
        # 기본 출력 폴더 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_output = os.path.join(script_dir, "output")
        self.output_folder.set(self.config.get("default_output_folder", default_output))
        
    def browse_folder(self):
        """폴더 선택 대화상자"""
        folder = filedialog.askdirectory(title="DOCX/PDF 파일이 있는 폴더를 선택하세요")
        if folder:
            self.selected_folder.set(folder)
            self.scan_files()
            
    def scan_files(self):
        """선택된 폴더에서 DOCX와 PDF 파일 검색"""
        folder = self.selected_folder.get()
        if not folder:
            return
            
        # 이전 파일 목록 지우기
        self.files_listbox.delete(0, tk.END)
        self.stats['files_list'] = []
        
        # DOCX와 PDF 파일 검색 (대소문자 구분 없이)
        files = []
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.docx', '.pdf')):
                files.append(os.path.join(folder, filename))
        
        # 파일 정렬
        files.sort()
        
        # UI 업데이트
        for file in files:
            filename = os.path.basename(file)
            self.files_listbox.insert(tk.END, filename)
            self.stats['files_list'].append(file)
        
        self.stats['total_files'] = len(files)
        self.files_label.config(text=f"0/{self.stats['total_files']}")
        
        self.log_message(f"처리할 파일 {len(files)}개를 찾았습니다")
        
    def browse_output_folder(self):
        """출력 폴더 선택 대화상자"""
        folder = filedialog.askdirectory(title="출력 폴더를 선택하세요")
        if folder:
            self.output_folder.set(folder)
            
    def start_processing(self):
        """일괄 처리 시작"""
        if not self.selected_folder.get():
            messagebox.showerror("오류", "먼저 폴더를 선택해 주세요")
            return
            
        if not self.stats['files_list']:
            messagebox.showerror("오류", "선택한 폴더에 DOCX/PDF 파일이 없습니다")
            return
            
        if not self.output_folder.get():
            messagebox.showerror("오류", "출력 폴더를 선택해 주세요")
            return
            
        # 통계 초기화
        self.stats['processed_files'] = 0
        self.stats['processed_formulas'] = 0
        self.stats['current_file'] = ''
        
        # UI 업데이트
        self.is_processing = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.open_folder_button.config(state="disabled")
        
        # 진행률 애니메이션 시작
        self.current_progress.config(mode='indeterminate')
        self.current_progress.start()
        
        # 처리 스레드 시작
        self.start_time = time.time()
        threading.Thread(target=self.process_files, daemon=True).start()
        
    def process_files(self):
        """모든 파일 처리"""
        try:
            for i, file_path in enumerate(self.stats['files_list']):
                if not self.is_processing:
                    break
                    
                filename = os.path.basename(file_path)
                self.stats['current_file'] = filename
                
                # UI 업데이트
                self.output_queue.put(('status', f"파일 처리 중 {i+1}/{self.stats['total_files']}: {filename}"))
                self.output_queue.put(('progress_file', filename))
                
                # 단일 파일 처리
                self.process_single_file(file_path)
                
                # 통계 업데이트
                self.stats['processed_files'] += 1
                self.output_queue.put(('progress_overall', (self.stats['processed_files'], self.stats['total_files'])))
                
                # 파일 간 짧은 대기
                time.sleep(0.5)
                
        except Exception as e:
            self.output_queue.put(('error', f"일괄 처리 오류: {str(e)}"))
        finally:
            self.output_queue.put(('finished', None))
            
    def process_single_file(self, file_path):
        """단일 파일 처리"""
        try:
            # 명령 준비
            script_dir = os.path.dirname(os.path.abspath(__file__))
            smartnougat_path = os.path.join(script_dir, "smartnougat_0715.py")
            
            # 명령 생성
            cmd = [sys.executable, smartnougat_path, file_path]
            
            # 페이지 범위 추가
            if self.page_range.get().strip():
                cmd.extend(["-p", self.page_range.get().strip()])
            
            # 출력 폴더 추가
            if self.output_folder.get():
                cmd.extend(["-o", self.output_folder.get()])
            
            # 명령 실행
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=script_dir
            )
            
            # 출력 읽기
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_processing:
                    break
                self.output_queue.put(('output', line.strip()))
                
            self.process.wait()
            
        except Exception as e:
            self.output_queue.put(('error', f"{os.path.basename(file_path)} 처리 중 오류: {str(e)}"))
            
    def stop_processing(self):
        """처리 중지"""
        self.is_processing = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
            
        self.current_progress.stop()
        self.current_progress.config(mode='determinate')
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.open_folder_button.config(state="normal")
        
        self.log_message("사용자가 처리를 중지했습니다")
        
    def start_output_reader(self):
        """출력 리더 시작"""
        self.process_output_queue()
        
    def process_output_queue(self):
        """출력 큐 처리"""
        try:
            while True:
                item = self.output_queue.get_nowait()
                msg_type, data = item
                
                if msg_type == 'output':
                    self.log_message(data)
                elif msg_type == 'error':
                    self.log_message(f"오류: {data}")
                elif msg_type == 'status':
                    self.log_message(data)
                elif msg_type == 'progress_file':
                    self.current_file_label.config(text=data)
                elif msg_type == 'progress_overall':
                    processed, total = data
                    self.files_label.config(text=f"{processed}/{total}")
                    if total > 0:
                        self.overall_progress.config(value=(processed/total)*100)
                elif msg_type == 'finished':
                    self.processing_finished()
                    
        except queue.Empty:
            pass
            
        # 다음 체크 예약
        self.root.after(100, self.process_output_queue)
        
    def processing_finished(self):
        """처리 완료 처리"""
        self.is_processing = False
        self.current_progress.stop()
        self.current_progress.config(mode='determinate', value=100)
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.open_folder_button.config(state="normal")
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        self.log_message(f"\n일괄 처리가 완료되었습니다!")
        self.log_message(f"처리된 파일: {self.stats['processed_files']}/{self.stats['total_files']}")
        self.log_message(f"총 처리 시간: {elapsed_time:.1f}초")
        
        # 완료 메시지 표시
        messagebox.showinfo("일괄 처리 완료", 
                          f"{self.stats['processed_files']}개 파일이 성공적으로 처리되었습니다!")
        
    def clear_log(self):
        """로그 텍스트 지우기"""
        self.log_text.delete(1.0, tk.END)
        
    def log_message(self, message):
        """로그에 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        
    def open_results_folder(self):
        """결과 폴더 열기"""
        if self.output_folder.get() and os.path.exists(self.output_folder.get()):
            os.startfile(self.output_folder.get())
        else:
            messagebox.showwarning("경고", "출력 폴더를 찾을 수 없습니다")
            
    def on_closing(self):
        """창 닫기 처리"""
        if self.is_processing:
            if messagebox.askokcancel("종료", "처리가 진행 중입니다. 중지하고 종료하시겠습니까?"):
                self.stop_processing()
            else:
                return
        
        self.save_config()
        self.root.destroy()

def main():
    """메인 함수"""
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        
    app = SmartNougatBatchGUI(root)
    
    # 창 닫기 처리
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.stop_processing()
        root.destroy()

if __name__ == "__main__":
    main()