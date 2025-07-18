#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartNougat 0712 - LaTeX 자동 수정 기능이 추가된 문서 처리 파이프라인
PDF/DOCX에서 수식을 추출하고 LaTeX로 변환 후 자동 수정
- 원본 출력 + 수정된 출력 병행 생성
- src/fix_latex.py를 통한 LaTeX 문법 자동 수정
- 수정된 버전의 별도 HTML 뷰어 생성
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Transformers 경고 메시지 숨기기
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from loguru import logger
from typing import List, Dict, Optional, Tuple

# Nougat 관련 imports
nougat_path = Path(__file__).parent / "nougat_latex"
if nougat_path.exists():
    sys.path.insert(0, str(nougat_path))
else:
    # Windows 경로 대체
    nougat_path = Path(__file__).parent / "nougat_latex"
    if nougat_path.exists():
        sys.path.insert(0, str(nougat_path))

try:
    from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoProcessor
    from transformers.models.nougat import NougatTokenizerFast
    # 로컬 util만 사용
    try:
        from nougat_latex.util import process_raw_latex_code
    except ImportError:
        # 간단한 대체 함수
        def process_raw_latex_code(latex_code):
            return latex_code.strip()
    
    # NougatLaTexProcessor import 추가
    try:
        from nougat_latex import NougatLaTexProcessor
        logger.info("NougatLaTexProcessor imported successfully")
    except ImportError:
        NougatLaTexProcessor = None
        logger.warning("NougatLaTexProcessor not available, using AutoProcessor")
    
    NOUGAT_AVAILABLE = True
    logger.info("Nougat LaTeX OCR available")
except ImportError as e:
    NOUGAT_AVAILABLE = False
    logger.error(f"Nougat을 사용할 수 없습니다: {e}")

# YOLO 관련 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLO available")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics가 설치되지 않았습니다. 수식 감지가 제한됩니다.")

# OCR 관련 imports (선택사항)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    logger.info("PaddleOCR available")
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR이 설치되지 않았습니다. OCR 기능이 제한됩니다.")

# DOCX 처리 (선택사항)
try:
    from docx2pdf import convert as docx2pdf_convert
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("docx2pdf가 설치되지 않았습니다. DOCX 지원이 비활성화됩니다.")

# Windows COM 지원 (선택사항)
try:
    import win32com.client
    import pythoncom
    WIN32COM_AVAILABLE = True
    logger.info("win32com available - DOCX page extraction supported")
except ImportError:
    WIN32COM_AVAILABLE = False
    logger.info("win32com이 없습니다. 전체 DOCX만 처리 가능")


class SmartNougatStandalone:
    """완전히 독립적인 Nougat 기반 문서 처리 파이프라인"""
    
    def __init__(self, device: str = 'auto', models_dir: Optional[str] = None):
        """
        SmartNougat 초기화
        
        Args:
            device: 'cuda', 'cpu', 또는 'auto' (자동 감지)
            models_dir: 모델 디렉토리 경로
        """
        # 디바이스 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"SmartNougat이 {self.device}에서 초기화됩니다")
        
        # 모델 디렉토리
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "..", "models", ".cache")
        
        # 모델 초기화
        self._init_models()
        
    def _init_models(self):
        """필요한 모든 모델 초기화"""
        logger.info("모델 로딩 중...")
        start_time = time.time()
        
        # 1. YOLO MFD 모델 (수식 감지)
        self.mfd_model = self._load_yolo_mfd()
        
        # 2. Nougat 모델 (수식 인식)
        self.nougat_model = self._load_nougat()
        
        # 3. OCR 모델 (선택사항)
        self.ocr_model = self._load_ocr()
        
        logger.info(f"모든 모델이 {time.time() - start_time:.2f}초 만에 로드되었습니다")
        
    def _load_yolo_mfd(self):
        """YOLO 기반 수식 감지 모델 로드"""
        if not YOLO_AVAILABLE:
            logger.error("YOLO를 사용할 수 없습니다. ultralytics를 설치하세요: pip install ultralytics")
            return None
            
        try:
            # PDF-Extract-Kit MFD 모델 경로들 시도
            possible_paths = [
                # 현재 디렉토리의 모델
                "pdf-extract-kit-models/models/MFD/YOLO/yolo_v8_ft.pt",
                "./pdf-extract-kit-models/models/MFD/YOLO/yolo_v8_ft.pt",
                # 사용자 홈의 모델
                os.path.expanduser("~/pdf-extract-kit-models/models/MFD/YOLO/yolo_v8_ft.pt"),
                # 캐시 디렉토리
                os.path.join(self.models_dir, "yolo_v8_formula_det_ft/weights/best.pt"),
                os.path.join(self.models_dir, "yolo_v8_formula_det_ft.pt"),
                # models 폴더의 YOLO 모델
                os.path.join(os.path.dirname(__file__), "..", "models", "yolo_v8_formula_det_ft.pt"),
                # Windows 경로들
                "C:/pdf-extract-kit-models/models/MFD/YOLO/yolo_v8_ft.pt",
                "C:/Users/Public/pdf-extract-kit-models/models/MFD/YOLO/yolo_v8_ft.pt"
            ]
            
            mfd_weight = None
            for path in possible_paths:
                if path.startswith("http"):
                    # 나중에 다운로드 구현
                    continue
                if os.path.exists(path):
                    mfd_weight = path
                    break
                    
            if mfd_weight and os.path.exists(mfd_weight):
                logger.info(f"MFD 모델 로딩: {mfd_weight}")
                return YOLO(mfd_weight)
            else:
                # 모델이 없으면 다운로드 안내
                logger.warning("MFD 모델을 찾을 수 없습니다.")
                logger.info("다음 명령으로 모델을 다운로드하세요:")
                logger.info("wget https://github.com/opendatalab/PDF-Extract-Kit/releases/download/PDFExtractv1.0/yolo_v8_formula_det_ft.pt")
                return None
                
        except Exception as e:
            logger.error(f"MFD 모델 로딩 실패: {e}")
            return None
            
    def _load_nougat(self):
        """Nougat 수식 인식 모델 로드"""
        if not NOUGAT_AVAILABLE:
            logger.error("Nougat을 사용할 수 없습니다!")
            return None
            
        try:
            # 로컬 모델 경로 사용
            local_model_path = os.path.join(os.path.dirname(__file__), "..", "models")
            
            # 먼저 로컬 경로에서 직접 로드 시도
            if os.path.exists(os.path.join(local_model_path, "config.json")):
                logger.info(f"Loading Nougat model from local path: {local_model_path}")
                model = VisionEncoderDecoderModel.from_pretrained(local_model_path)
                tokenizer = NougatTokenizerFast.from_pretrained(local_model_path)
                
                # processor 시도
                processor = None
                try:
                    if NougatLaTexProcessor is not None:
                        processor = NougatLaTexProcessor.from_pretrained(local_model_path)
                        logger.info("Using NougatLaTexProcessor")
                    else:
                        processor = AutoProcessor.from_pretrained(local_model_path)
                        logger.info("Using AutoProcessor")
                except:
                    logger.warning("Processor not found, using tokenizer only")
                
                model.to(self.device)
                model.eval()
                
                return {
                    'model': model,
                    'tokenizer': tokenizer,
                    'processor': processor,
                    'type': 'local'
                }
            
            # 캐시에서 로드 시도
            model_name = "Norm/nougat-latex-base"
            
            # 오프라인 모드로 시도
            try:
                logger.info(f"Attempting offline loading of {model_name}...")
                model = VisionEncoderDecoderModel.from_pretrained(
                    model_name, 
                    local_files_only=True,
                    cache_dir=self.models_dir
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                    cache_dir=self.models_dir
                )
                # Processor가 없으면 None으로 설정
                processor = None
                if 'NougatLaTexProcessor' in globals():
                    processor = NougatLaTexProcessor.from_pretrained(
                        model_name,
                        local_files_only=True
                    )
                logger.info("Successfully loaded model in offline mode")
                
            except Exception as offline_error:
                logger.warning(f"Offline loading failed: {offline_error}")
                logger.info("Attempting online loading...")
                
                # 온라인 모드로 시도
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
                tokenizer = NougatTokenizerFast.from_pretrained(model_name)
                processor = None
                if 'NougatLaTexProcessor' in globals():
                    processor = NougatLaTexProcessor.from_pretrained(model_name)
                logger.info("Successfully loaded model from online")
            
            model.to(self.device)
            model.eval()
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'processor': processor,
                'type': 'transformers'
            }
            
        except Exception as e:
            logger.error(f"Nougat 모델 로딩 실패: {e}")
            logger.error("Please run 'python offline_model_setup.py' to download models for offline use")
            return None
            
    def _load_ocr(self):
        """OCR 모델 로드 (선택사항)"""
        if not PADDLE_AVAILABLE:
            logger.info("PaddleOCR을 사용할 수 없습니다. OCR 기능이 제한됩니다.")
            return None
            
        try:
            # PaddleOCR 초기화
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=(self.device == 'cuda')
            )
            return ocr
            
        except Exception as e:
            logger.warning(f"OCR 모델 로딩 실패: {e}")
            return None
            
    def process_document(self, input_path: str, output_dir: str, 
                        page_range: Optional[str] = None) -> Dict:
        """
        문서 처리 메인 함수
        
        Args:
            input_path: 입력 파일 경로 (PDF/DOCX)
            output_dir: 출력 디렉토리
            page_range: 페이지 범위 (예: "1-5", "3,5,7")
            
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        input_path = Path(input_path)
        
        # 입력 검증
        if not input_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
            
        # 출력 디렉토리 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(output_dir) / f"{input_path.stem}_smartnougat_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 설정
        log_file = output_path / "processing.log"
        logger.add(log_file, rotation="10 MB", encoding="utf-8")
        
        logger.info(f"Starting document processing: {input_path}")
        logger.info(f"Output directory: {output_path}")
        
        # DOCX → PDF 변환 (필수)
        if input_path.suffix.lower() == '.docx':
            # 먼저 전체 DOCX를 PDF로 변환
            if not DOCX_AVAILABLE:
                raise ImportError("DOCX 처리를 위해 docx2pdf를 설치하세요: pip install docx2pdf")
            pdf_path = self._convert_docx_to_pdf(input_path, output_path)
            
            # 페이지 범위가 있으면 추출
            if page_range:
                pdf_path = self._extract_pages(pdf_path, page_range, output_path)
        else:
            pdf_path = input_path
            
            # PDF의 페이지 범위 처리
            if page_range:
                pdf_path = self._extract_pages(pdf_path, page_range, output_path)
            
        # PDF 처리
        result = self._process_pdf(pdf_path, output_path)
        
        # 처리 시간
        result['processing_time'] = time.time() - start_time
        
        # 요약 저장
        self._save_processing_summary(result, output_path)
        
        logger.info(f"Document processing complete: {result['processing_time']:.2f}s")
        
        return result
        
    def _process_pdf(self, pdf_path: Path, output_path: Path) -> Dict:
        """PDF 처리 핵심 로직"""
        logger.info("Starting PDF processing...")
        
        # PyMuPDF 캐시 초기화
        fitz.TOOLS.store_shrink(100)  # 캐시 크기를 100%로 축소 (모두 삭제)
        
        # PDF 열기
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        
        # 디렉토리 구조 생성
        dirs = self._create_directory_structure(output_path)
        
        # 결과 저장용
        all_pages_data = []
        all_formulas = []
        
        # 페이지별 처리
        for page_num in range(total_pages):
            logger.info(f"Processing page {page_num + 1}/{total_pages}... [{(page_num + 1) / total_pages * 100:.1f}%]")
            
            page = pdf_doc[page_num]
            page_data = self._process_single_page(page, page_num, dirs)
            
            all_pages_data.append(page_data)
            all_formulas.extend(page_data.get('formulas', []))
            
            # 메모리 관리 - 매 3페이지마다 캐시 정리 (개선)
            # PyMuPDF는 렌더링된 페이지를 메모리에 캐시로 보관
            # 대용량 PDF 처리시 메모리 부족 방지를 위해 주기적으로 정리
            if (page_num + 1) % 3 == 0:
                fitz.TOOLS.store_shrink(100)  # 캐시 전체 제거
                import gc
                gc.collect()  # 가비지 커렉션 강제 실행
                logger.info(f"Cache and memory cleanup complete (page {page_num + 1}/{total_pages})")
            
        # 결과 저장
        self._save_results(all_pages_data, all_formulas, output_path)
        
        # HTML 뷰어 생성 - 제거됨 (src/create_viewer.py 사용)
        
        pdf_doc.close()
        
        return {
            'success': True,
            'output_dir': str(output_path),
            'pages': total_pages,
            'total_formulas': len(all_formulas),
            'formula_details': all_formulas
        }
        
    def _create_directory_structure(self, output_path: Path) -> Dict[str, Path]:
        """출력 디렉토리 구조 생성"""
        dirs = {
            'pages': output_path / 'pages',
            'images': output_path / 'images',  # 수식, 표, 일반 이미지 모두 여기에
            'txt': output_path / 'txt',
            'txt_images': output_path / 'txt' / 'images'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return dirs
        
    def _process_single_page(self, page, page_num: int, dirs: Dict[str, Path]) -> Dict:
        """단일 페이지 처리"""
        # 페이지를 이미지로 변환
        mat = fitz.Matrix(2, 2)  # 2배 확대
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # 페이지 이미지 저장
        page_img_path = dirs['pages'] / f"page_{page_num}.png"
        img.save(page_img_path)
        
        # 수식 감지
        formulas = self._detect_formulas(img_array, page_num)
        
        # 수식 이미지 추출 및 LaTeX 변환
        for idx, formula in enumerate(formulas):
            # bbox 확장
            expanded_bbox = self._expand_bbox(
                formula['bbox'], 
                img_array.shape, 
                expand_ratio_x=0.15, 
                expand_ratio_y=0.03
            )
            
            # 수식 이미지 추출
            formula_img = self._extract_image_region(img_array, expanded_bbox)
            
            # 이미지 저장
            formula_filename = f"formula_page{page_num}_{idx:03d}.png"
            formula_path = dirs['images'] / formula_filename
            Image.fromarray(formula_img).save(formula_path)
            logger.info(f"Processing formula {idx + 1}/{len(formulas)}: {formula_filename}")
            
            # Nougat으로 LaTeX 변환
            latex = self._recognize_formula_with_nougat(formula_img)
            
            # 정보 업데이트
            formula['image_path'] = str(formula_path)
            formula['latex'] = latex
            formula['page_num'] = page_num
            formula['index'] = idx
            
        # 텍스트 추출 (OCR 또는 PDF 텍스트)
        text_blocks = self._extract_text(page, img_array)
        
        return {
            'page_num': page_num,
            'page_size': [pix.width, pix.height],
            'formulas': formulas,
            'text_blocks': text_blocks,
            'page_image': str(page_img_path)
        }
        
    def _detect_formulas(self, img_array: np.ndarray, page_num: int) -> List[Dict]:
        """수식 위치 감지"""
        formulas = []
        
        if self.mfd_model is not None:
            # YOLO MFD 사용
            results = self.mfd_model.predict(
                img_array, 
                imgsz=1888, 
                conf=0.25, 
                iou=0.45, 
                verbose=False
            )[0]
            
            for idx, (xyxy, conf, cls) in enumerate(
                zip(results.boxes.xyxy.cpu(), 
                    results.boxes.conf.cpu(), 
                    results.boxes.cls.cpu())
            ):
                x1, y1, x2, y2 = [int(p.item()) for p in xyxy]
                
                formula = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'category': 'inline' if cls == 0 else 'block',
                    'category_id': 13 if cls == 0 else 14
                }
                
                formulas.append(formula)
                
        else:
            # MFD 모델이 없으면 수식을 감지할 수 없음
            logger.error("MFD model not found. Cannot detect formulas.")
            logger.info("Download the model with this command:")
            logger.info("wget https://github.com/opendatalab/PDF-Extract-Kit/releases/download/PDFExtractv1.0/yolo_v8_formula_det_ft.pt")
                
        logger.info(f"Detected {len(formulas)} formulas on page {page_num}")
        return formulas
        
    def _expand_bbox(self, bbox: List[int], img_shape: Tuple, 
                     expand_ratio_x: float = 0.25, 
                     expand_ratio_y: float = 0.03) -> List[int]:
        """bbox 확장 (25% 너비, 3% 높이)"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 확장 크기 계산
        expand_x = int(width * expand_ratio_x)
        expand_y = int(height * expand_ratio_y)
        
        # 확장된 bbox (이미지 경계 체크)
        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = min(img_shape[1], x2 + expand_x)
        new_y2 = min(img_shape[0], y2 + expand_y)
        
        return [new_x1, new_y1, new_x2, new_y2]
        
    def _extract_image_region(self, img_array: np.ndarray, bbox: List[int]) -> np.ndarray:
        """이미지 영역 추출"""
        x1, y1, x2, y2 = bbox
        
        # 이미지 경계 체크
        h, w = img_array.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h))
        
        # 유효한 영역인지 확인
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"잘못된 bbox: [{x1}, {y1}, {x2}, {y2}]")
            # 최소한의 크기 반환
            return img_array[0:10, 0:10]
            
        return img_array[y1:y2, x1:x2]
        
    def _recognize_formula_with_nougat(self, formula_img: np.ndarray) -> str:
        """Nougat으로 수식 인식"""
        if self.nougat_model is None:
            logger.warning("Nougat 모델이 없습니다")
            return ""
            
        try:
            # numpy array를 PIL Image로 변환
            if isinstance(formula_img, np.ndarray):
                formula_img = Image.fromarray(formula_img)
                
            # RGB로 변환
            if formula_img.mode != "RGB":
                formula_img = formula_img.convert('RGB')
            
            # 디버깅: 이미지 크기 확인
            logger.info(f"Formula image size: {formula_img.size}")
            
            # transformers 방식 (로컬 또는 캐시)
            processor = self.nougat_model.get('processor')
            model = self.nougat_model['model']
            tokenizer = self.nougat_model['tokenizer']
            
            # processor가 없으면 직접 처리
            if processor is None:
                # 간단한 전처리
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = transform(formula_img).unsqueeze(0)
            else:
                pixel_values = processor(formula_img, return_tensors="pt").pixel_values
            
            task_prompt = tokenizer.bos_token
            decoder_input_ids = tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            
            # 생성
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values.to(self.device),
                    decoder_input_ids=decoder_input_ids.to(self.device),
                    max_length=model.decoder.config.max_length,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
                
            # 디코딩
            sequence = tokenizer.batch_decode(outputs.sequences)[0]
            logger.info(f"Raw Nougat output: {sequence[:100]}...")  # 처음 100자 출력
            
            sequence = sequence.replace(tokenizer.eos_token, "").replace(
                tokenizer.pad_token, "").replace(tokenizer.bos_token, "")
            
            if 'process_raw_latex_code' in globals():
                sequence = process_raw_latex_code(sequence)
            
            logger.info(f"Processed LaTeX: {sequence[:50]}...")  # 처리된 LaTeX 확인
            return sequence.strip()
                
        except Exception as e:
            logger.error(f"Nougat 인식 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
            
    def _extract_text(self, page, img_array: np.ndarray) -> List[Dict]:
        """텍스트 추출"""
        text_blocks = []
        
        # 먼저 PDF에서 직접 텍스트 추출 시도
        try:
            # UTF-8 인코딩으로 텍스트 추출
            text = page.get_text("text", flags=11)  # preserve ligatures, preserve whitespace
            if text.strip():
                # 텍스트를 라인별로 분리
                lines = text.split('\n')
                for line in lines:
                    if line.strip():
                        text_blocks.append({
                            'type': 'text',
                            'content': line.strip(),
                            'source': 'pdf'
                        })
                        
                # 텍스트가 있으면 OCR 건너뛰기
                if text_blocks:
                    return text_blocks
        except Exception as e:
            logger.warning(f"PDF 텍스트 추출 실패: {e}")
            
        # OCR 사용 (가능한 경우)
        if self.ocr_model is not None:
            try:
                result = self.ocr_model.ocr(img_array, cls=True)
                for line in result:
                    if line:
                        for box, (text, conf) in line:
                            text_blocks.append({
                                'type': 'text',
                                'content': text,
                                'confidence': conf,
                                'bbox': box,
                                'source': 'ocr'
                            })
            except Exception as e:
                logger.warning(f"OCR 실패: {e}")
                
        return text_blocks
        
    def _save_results(self, pages_data: List[Dict], formulas: List[Dict], output_path: Path):
        """결과 저장"""
        # model.json 형식으로 저장
        model_data = []
        for page_data in pages_data:
            page_model = {
                'page_idx': page_data['page_num'],
                'page_size': page_data['page_size'],
                'layout_dets': []
            }
            
            # 수식 정보 추가
            for formula in page_data.get('formulas', []):
                det = {
                    'category_id': formula['category_id'],
                    'poly': self._bbox_to_poly(formula['bbox']),
                    'score': formula['confidence'],
                    'latex': formula['latex']
                }
                page_model['layout_dets'].append(det)
                
            model_data.append(page_model)
            
        # model.json 저장
        model_path = output_path / 'txt' / 'model.json'
        model_path.parent.mkdir(exist_ok=True)
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
            
        # middle.json 저장
        middle_data = {
            'pdf_info': pages_data,
            'model_list': model_data,
            'pdf_type': 'txt',
            '_pdf_type': 'txt'
        }
        
        middle_path = output_path / 'txt' / 'middle.json'
        with open(middle_path, 'w', encoding='utf-8') as f:
            json.dump(middle_data, f, ensure_ascii=False, indent=2)
            
        # 간단한 마크다운 파일도 생성 (Universal 뷰어를 위해)
        md_content = self._generate_markdown(pages_data)
        md_path = output_path / 'txt' / f'{output_path.name}.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        # Layout PDF 생성 (선택사항)
        self._generate_layout_pdf(pages_data, output_path)
            
        logger.info(f"결과가 저장되었습니다: {output_path}")
        
    def _generate_markdown(self, pages_data: List[Dict]) -> str:
        """페이지 데이터에서 마크다운 생성"""
        md_lines = []
        
        for page_data in pages_data:
            page_num = page_data.get('page_num', 0)
            md_lines.append(f"\n## Page {page_num + 1}\n")
            
            # 텍스트와 수식을 함께 표시
            for item in page_data.get('text_blocks', []):
                if item.get('content'):
                    md_lines.append(item['content'] + "\n")
                    
            for idx, formula in enumerate(page_data.get('formulas', [])):
                if formula.get('category') == 'inline':
                    md_lines.append(f"${formula.get('latex', '')}$")
                else:
                    md_lines.append(f"\n$$\n{formula.get('latex', '')}\n$$\n")
                    
        return '\n'.join(md_lines)
        
    def _bbox_to_poly(self, bbox: List[int]) -> List[int]:
        """bbox를 polygon 형식으로 변환"""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2, y1, x2, y2, x1, y2]
        
    def _generate_html_viewer(self, pages_data: List[Dict], pdf_path: Path, output_path: Path):
        """HTML 뷰어 생성 - 제거됨"""
        # 기본 HTML 뷰어 생성 제거
        # src/create_viewer.py를 사용하여 result_viewer_0715.html 생성
        pass
            
    def _create_simple_html_viewer(self, pages_data: List[Dict], pdf_path: Path, output_path: Path):
        """간단한 HTML 뷰어 생성 - 제거됨"""
        # 기본 HTML 뷰어 생성 제거
        pass
        
    def _save_processing_summary(self, result: Dict, output_path: Path):
        """처리 요약 저장"""
        summary = {
            'processing_time': result['processing_time'],
            'total_pages': result['pages'],
            'total_formulas': result['total_formulas'],
            'output_directory': str(output_path),
            'timestamp': datetime.now().isoformat(),
            'formulas': result['formula_details']
        }
        
        summary_path = output_path / 'processing_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
    def _extract_docx_pages_with_com(self, docx_path: Path, page_range: str, output_path: Path) -> Path:
        """win32com을 사용하여 DOCX에서 특정 페이지 추출"""
        if not WIN32COM_AVAILABLE:
            logger.warning("win32com이 없어서 전체 문서를 변환합니다")
            return None
            
        try:
            pythoncom.CoInitialize()
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            
            # 문서 열기
            doc = word.Documents.Open(str(docx_path.absolute()))
            
            # 페이지 범위 파싱
            pages = self._parse_page_range_for_word(page_range, doc)
            
            # 새 문서 생성
            new_doc = word.Documents.Add()
            
            # 페이지별로 복사
            for page_num in pages:
                # 페이지 선택
                doc.Activate()
                word.Selection.GoTo(What=1, Which=1, Count=page_num)  # wdGoToPage=1
                word.Selection.GoTo(What=1, Which=1, Count=page_num+1)
                word.Selection.MoveUp()
                word.Selection.Extend()
                word.Selection.GoTo(What=1, Which=1, Count=page_num)
                
                # 복사 및 붙여넣기
                word.Selection.Copy()
                new_doc.Activate()
                word.Selection.Paste()
                
                # 페이지 나누기 추가 (마지막 페이지 제외)
                if page_num != pages[-1]:
                    word.Selection.InsertBreak(7)  # wdPageBreak
            
            # PDF로 저장
            pdf_path = output_path / f"{docx_path.stem}_pages_{page_range.replace(':', '-')}.pdf"
            new_doc.SaveAs2(str(pdf_path.absolute()), FileFormat=17)  # wdFormatPDF=17
            
            # 정리
            new_doc.Close()
            doc.Close()
            word.Quit()
            pythoncom.CoUninitialize()
            
            logger.info(f"DOCX 페이지 추출 완료: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"win32com 페이지 추출 실패: {e}")
            if 'word' in locals():
                word.Quit()
            pythoncom.CoUninitialize()
            return None
            
    def _parse_page_range_for_word(self, page_range: str, doc) -> List[int]:
        """Word 문서의 페이지 범위 파싱"""
        # 전체 페이지 수 가져오기
        doc.Repaginate()
        total_pages = doc.ComputeStatistics(2)  # wdStatisticPages=2
        
        pages = []
        page_range = page_range.strip()
        
        if '-' in page_range and ',' not in page_range:
            parts = page_range.split('-')
            if len(parts) == 2:
                if parts[0] and parts[1]:
                    start, end = int(parts[0]), int(parts[1])
                    pages = list(range(start, end+1))
                elif parts[0]:
                    start = int(parts[0])
                    pages = list(range(start, total_pages+1))
                elif parts[1]:
                    end = int(parts[1])
                    pages = list(range(1, end+1))
        elif ',' in page_range:
            pages = [int(p.strip()) for p in page_range.split(',') if p.strip()]
        elif ':' in page_range:
            start, end = map(int, page_range.split(':'))
            pages = list(range(start, end+1))
        else:
            pages = [int(page_range)]
            
        # 유효한 페이지만 필터링
        return [p for p in pages if 1 <= p <= total_pages]
    
    def _generate_layout_pdf(self, pages_data: List[Dict], output_path: Path):
        """레이아웃 분석 결과를 시각화한 PDF 생성"""
        try:
            # 원본 PDF 경로 찾기
            pdf_files = list(output_path.glob("*.pdf"))
            if not pdf_files:
                return
                
            source_pdf = None
            for pdf in pdf_files:
                # pages_가 있는 파일을 우선적으로 선택 (추출된 페이지)
                if "pages_" in pdf.name:
                    source_pdf = pdf
                    break
            
            # pages_가 없으면 원본 PDF 사용
            if not source_pdf:
                for pdf in pdf_files:
                    if pdf.name.endswith('.pdf') and "layout" not in pdf.name:
                        source_pdf = pdf
                        break
                    
            if not source_pdf:
                return
                
            # PDF 열기
            doc = fitz.open(source_pdf)
            
            # 각 페이지에 박스 그리기
            for page_data in pages_data:
                page_num = page_data.get('page_num', 0)
                if page_num >= len(doc):
                    continue
                    
                page = doc[page_num]
                
                # 수식 박스 그리기
                for formula in page_data.get('formulas', []):
                    bbox = formula.get('bbox', [])
                    if len(bbox) == 4:
                        # 2배 확대된 좌표를 원본 크기로 변환
                        x1, y1, x2, y2 = bbox
                        rect = fitz.Rect(x1/2, y1/2, x2/2, y2/2)
                        # 인라인 수식: 파란색, 블록 수식: 빨간색
                        if formula.get('category_id') == 13:  # inline
                            color = (0, 0, 1)  # 파란색
                        else:  # block
                            color = (1, 0, 0)  # 빨간색
                        page.draw_rect(rect, color=color, width=2)
                        
                # 텍스트 블록 박스 그리기 (녹색)
                for text_block in page_data.get('text_blocks', []):
                    if 'bbox' in text_block and text_block['bbox']:
                        bbox = text_block['bbox']
                        if isinstance(bbox[0], list):  # OCR bbox 형식
                            x_coords = [p[0]/2 for p in bbox]
                            y_coords = [p[1]/2 for p in bbox]
                            rect = fitz.Rect(min(x_coords), min(y_coords), 
                                           max(x_coords), max(y_coords))
                        else:
                            # 2배 확대된 좌표를 원본 크기로 변환
                            x1, y1, x2, y2 = bbox
                            rect = fitz.Rect(x1/2, y1/2, x2/2, y2/2)
                        page.draw_rect(rect, color=(0, 1, 0), width=1)
            
            # 저장
            layout_pdf_path = output_path / "layout.pdf"
            doc.save(str(layout_pdf_path))
            doc.close()
            
            logger.info(f"Layout PDF 생성: {layout_pdf_path}")
            
        except Exception as e:
            logger.warning(f"Layout PDF 생성 실패: {e}")
    
    def _convert_docx_to_pdf(self, docx_path: Path, output_path: Path) -> Path:
        """DOCX를 PDF로 변환"""
        # 출력 디렉토리 확인 및 생성
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_path / f"{docx_path.stem}.pdf"
        logger.info(f"DOCX → PDF 변환: {docx_path} → {pdf_path}")
        
        # win32com 사용 시도 (수식 보존이 더 좋음)
        if WIN32COM_AVAILABLE:
            try:
                pythoncom.CoInitialize()
                
                # 항상 새로운 Word 인스턴스 생성 (기존 작업에 영향 없음)
                word = win32com.client.DispatchEx("Word.Application")  # DispatchEx는 항상 새 인스턴스 생성
                logger.debug("독립적인 Word 인스턴스 생성")
                
                word.Visible = False  # 보이지 않게 설정
                word.DisplayAlerts = False  # 경고 대화상자 비활성화
                word.ScreenUpdating = False  # 화면 업데이트 비활성화 (성능 향상)
                
                # Windows 경로로 변환
                docx_win_path = str(docx_path.absolute()).replace('/', '\\')
                pdf_win_path = str(pdf_path.absolute()).replace('/', '\\')
                
                logger.debug(f"DOCX 경로: {docx_win_path}")
                logger.debug(f"PDF 경로: {pdf_win_path}")
                
                # 문서 열기
                doc = word.Documents.Open(docx_win_path)
                
                if doc is None:
                    raise Exception("문서를 열 수 없습니다")
                
                # PDF로 저장 (wdFormatPDF = 17)
                doc.SaveAs2(pdf_win_path, FileFormat=17)
                doc.Close(False)  # 저장하지 않고 닫기
                
                # Word 종료 (항상 종료 - 독립 인스턴스이므로)
                try:
                    word.Quit(SaveChanges=False)  # 변경사항 저장하지 않고 종료
                except:
                    pass  # 종료 실패해도 계속 진행
                
                pythoncom.CoUninitialize()
                logger.info("win32com으로 DOCX → PDF 변환 성공")
                return pdf_path
                
            except Exception as e:
                import traceback
                logger.warning(f"win32com 변환 실패: {e}")
                logger.debug(f"상세 오류: {traceback.format_exc()}")
                # docx2pdf로 폴백
        
        # docx2pdf 사용 (폴백)
        if not DOCX_AVAILABLE:
            raise ImportError("docx2pdf가 설치되지 않았습니다. pip install docx2pdf")
            
        docx2pdf_convert(str(docx_path), str(pdf_path))
        logger.info("docx2pdf로 DOCX → PDF 변환 완료")
        
        return pdf_path
        
    def _extract_pages(self, pdf_path: Path, page_range: str, output_path: Path) -> Path:
        """페이지 추출"""
        # 페이지 범위 파싱
        pages = []
        
        try:
            # 공백 제거
            page_range = page_range.strip()
            
            if '-' in page_range and ',' not in page_range:
                # 범위 형식: "1-5" 또는 "-5" 또는 "5-"
                parts = page_range.split('-')
                if len(parts) == 2:
                    if parts[0] and parts[1]:  # "1-5"
                        start, end = int(parts[0]), int(parts[1])
                        pages = list(range(start-1, end))
                    elif parts[0]:  # "5-" (5페이지부터 끝까지)
                        start = int(parts[0])
                        doc_temp = fitz.open(pdf_path)
                        pages = list(range(start-1, len(doc_temp)))
                        doc_temp.close()
                    elif parts[1]:  # "-5" (처음부터 5페이지까지)
                        end = int(parts[1])
                        pages = list(range(0, end))
            elif ',' in page_range:
                # 콤마 구분 형식: "1,3,5"
                pages = [int(p.strip())-1 for p in page_range.split(',') if p.strip()]
            elif ':' in page_range:
                # 콜론 형식 처리: "1:20", ":20", "1:"
                parts = page_range.split(':')
                if len(parts) == 2:
                    if parts[0] and parts[1]:  # "1:20"
                        start, end = int(parts[0]), int(parts[1])
                        pages = list(range(start-1, end))
                    elif parts[0]:  # "1:" (1페이지부터 끝까지)
                        start = int(parts[0])
                        doc_temp = fitz.open(pdf_path)
                        pages = list(range(start-1, len(doc_temp)))
                        doc_temp.close()
                    elif parts[1]:  # ":20" (처음부터 20페이지까지)
                        end = int(parts[1])
                        pages = list(range(0, end))
            else:
                # 단일 페이지
                pages = [int(page_range)-1]
                
        except Exception as e:
            logger.error(f"페이지 범위 파싱 오류: {e}")
            raise ValueError(f"잘못된 페이지 범위 형식: {page_range}. 예: '1-5', '1,3,5', '10'.")
            
        # 페이지 추출
        doc = fitz.open(pdf_path)
        new_doc = fitz.open()
        
        for page_num in pages:
            if 0 <= page_num < len(doc):
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
        # 저장 - PDF/A 형식으로 저장하여 폰트 포함
        extracted_path = output_path / f"{pdf_path.stem}_pages_{page_range}.pdf"
        # 폰트 포함을 위한 옵션 설정
        new_doc.save(
            str(extracted_path),
            garbage=4,  # 최대 압축
            clean=True,  # 불필요한 객체 제거
            deflate=True,  # 압축 사용
            deflate_images=True,  # 이미지 압축
            deflate_fonts=True  # 폰트 압축
        )
        
        doc.close()
        new_doc.close()
        
        return extracted_path


def create_fixed_md(txt_dir):
    """Create output_fixed.md from model_fixed.json"""
    model_fixed_path = txt_dir / "model_fixed.json"
    output_md_path = txt_dir / "output.md"
    output_fixed_md_path = txt_dir / "output_fixed.md"
    
    # Read fixed model data
    with open(model_fixed_path, 'r', encoding='utf-8') as f:
        fixed_data = json.load(f)
    
    # Read original markdown if exists
    if output_md_path.exists():
        with open(output_md_path, 'r', encoding='utf-8') as f:
            original_md = f.read()
    else:
        # Create new markdown if original doesn't exist
        original_md = ""
        for item in fixed_data:
            original_md += f"\n$$\n{item.get('latex_original', item.get('latex', ''))}\n$$\n\n"
    
    # Create a mapping of original LaTeX to fixed LaTeX
    latex_mapping = {}
    for item in fixed_data:
        if 'latex' in item:  # Check if latex key exists
            if item.get('latex_original'):
                # If there was a fix, map original to fixed
                latex_mapping[item['latex_original']] = item['latex']
            else:
                # If no fix, keep the same
                latex_mapping[item['latex']] = item['latex']
    
    # Replace LaTeX in markdown
    fixed_md = original_md
    for original, fixed in latex_mapping.items():
        if original and fixed and original != fixed:
            # Replace $$original$$ with $$fixed$$
            fixed_md = fixed_md.replace(f"$$\n{original}\n$$", f"$$\n{fixed}\n$$")
            fixed_md = fixed_md.replace(f"$${original}$$", f"$${fixed}$$")
            # Also try with inline math
            fixed_md = fixed_md.replace(f"${original}$", f"${fixed}$")
    
    # Save fixed markdown
    with open(output_fixed_md_path, 'w', encoding='utf-8') as f:
        f.write(fixed_md)
    
    return output_fixed_md_path


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="SmartNougat Standalone - 독립 실행형 문서 처리"
    )
    parser.add_argument('input', help='입력 파일 경로 (PDF/DOCX)')
    parser.add_argument('-o', '--output', default='./output', help='출력 디렉토리')
    parser.add_argument('-p', '--pages', help='페이지 범위 (예: 1-5 또는 1,3,5)')
    parser.add_argument('--local-mathjax', action='store_true', help='로컬 MathJax 사용 (오프라인 모드)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    
    args = parser.parse_args()
    
    # 로거 설정
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
        
    # SmartNougat 실행
    try:
        processor = SmartNougatStandalone(device=args.device)
        result = processor.process_document(
            args.input,
            args.output,
            page_range=args.pages
        )
        
        # 결과 출력
        print(f"\n[SUCCESS] Processing complete!")
        print(f"[OUTPUT] Directory: {result['output_dir']}")
        print(f"[PAGES] Total: {result['pages']}")
        print(f"[FORMULAS] Found {result['total_formulas']} total")
        print(f"[TIME] Processing time: {result['processing_time']:.2f}s")
        
        # LaTeX 수정 처리
        print(f"\n[POST-PROCESSING] Fixing LaTeX syntax...")
        txt_dir = Path(result['output_dir']) / "txt"
        model_json_path = txt_dir / "model.json"
        
        if model_json_path.exists():
            try:
                # fix_latex.py 실행
                import subprocess
                fix_latex_path = os.path.join(os.path.dirname(__file__), "fix_latex.py")
                fix_result = subprocess.run(
                    [sys.executable, fix_latex_path, str(model_json_path)],
                    capture_output=True,
                    text=True
                )
                
                if fix_result.returncode == 0:
                    print("[✓] LaTeX fixing complete")
                    
                    # output_fixed.md 생성
                    model_fixed_path = txt_dir / "model_fixed.json"
                    if model_fixed_path.exists():
                        create_fixed_md(txt_dir)
                        print("[✓] output_fixed.md created")
                        
                        # Fixed HTML viewer 생성 - 0715 버전 사용
                        create_viewer_path = os.path.join(os.path.dirname(__file__), "create_viewer.py")
                        viewer_cmd = [sys.executable, create_viewer_path, str(result['output_dir'])]
                        
                        # 사용자가 명시적으로 옵션을 지정한 경우에만 추가
                        if hasattr(args, 'local_mathjax') and args.local_mathjax:
                            viewer_cmd.append("--local-mathjax")
                        # 기본값은 자동 감지이므로 아무것도 추가하지 않음
                        
                        viewer_result = subprocess.run(
                            viewer_cmd,
                            capture_output=True,
                            text=True
                        )
                        
                        if viewer_result.returncode == 0:
                            print("[✓] result_viewer_0715.html created")
                        else:
                            print(f"[경고] Fixed viewer 생성 실패: {viewer_result.stderr}")
                else:
                    print(f"[경고] LaTeX 수정 실패: {fix_result.stderr}")
                    
            except Exception as e:
                print(f"[경고] 추가 처리 중 오류: {e}")
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()