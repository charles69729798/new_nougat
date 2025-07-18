#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create simple vertical layout HTML viewer for fixed LaTeX results
이미지 → 원본 LaTeX → 검증된 LaTeX → 렌더링 순서로 세로 배치
"""

import json
import html
import base64
import sys
import os
from pathlib import Path
from datetime import datetime

# fix_latex 모듈 경로를 sys.path에 추가
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import fix_latex


def create_fixed_viewer(output_dir, scale=1.5):
    """Create HTML viewer for fixed LaTeX results - simple vertical layout"""
    
    output_dir = Path(output_dir)
    txt_dir = output_dir / "txt"
    
    # Read original JSON only (fix_latex will be applied in real-time)
    model_json_path = txt_dir / "model.json"
    
    if not model_json_path.exists():
        print(f"Error: {model_json_path} not found")
        return False
    
    with open(model_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # LaTeXFixer 인스턴스 생성하여 실시간으로 fix_latex 적용
    fixer = fix_latex.LaTeXFixer()
    
    # Process data
    formula_results = []
    fixed_count = 0
    total_formulas = 0
    formula_index = 0
    
    for page_idx, orig_page in enumerate(original_data):
        for det_idx, orig_det in enumerate(orig_page.get('layout_dets', [])):
            if 'latex' in orig_det:
                total_formulas += 1
                
                orig_latex = orig_det.get('latex', '')
                
                # fix_latex.py를 직접 실행하여 최신 패턴 적용
                fixed_latex, fixes = fixer.fix_latex_code(orig_latex)
                
                # Check if latex was fixed
                was_fixed = orig_latex != fixed_latex
                if was_fixed:
                    fixed_count += 1
                
                formula_results.append({
                    'index': formula_index,
                    'page_index': page_idx,
                    'original_latex': orig_latex,
                    'fixed_latex': fixed_latex,
                    'was_fixed': was_fixed,
                    'category_id': orig_det.get('category_id', ''),
                    'filename': f"formula_page{page_idx}_{det_idx:03d}.png",
                    'fixes': fixes  # 어떤 수정이 적용되었는지 기록
                })
                formula_index += 1
    
    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartNougat LaTeX Results</title>
    
    <!-- MathJax -->
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true
            }},
            svg: {{
                fontCache: 'global',
                scale: {scale * 0.92}
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    
    <style>
        body {{
            font-family: -apple-system, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .formula-card {{
            background: white;
            margin-bottom: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e5e7eb;
            font-size: 16px;
        }}
        
        .formula-number {{
            font-weight: bold;
            color: #2563eb;
        }}
        
        .category-inline {{
            color: #10b981;
            font-weight: 500;
        }}
        
        .category-block {{
            color: #8b5cf6;
            font-weight: 500;
        }}
        
        .fixed {{
            color: #10b981;
            font-weight: 600;
        }}
        
        /* 이미지 섹션 */
        .formula-image {{
            text-align: center;
            margin-bottom: 25px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            position: relative;
            overflow: auto;
        }}
        
        .formula-image img {{
            max-width: 100%;
            height: auto;
            display: inline-block;
            transform-origin: center;
            transition: transform 0.2s ease;
            transform: scale(1.2);
        }}
        
        .formula-image .zoom-controls {{
            background: rgba(37, 99, 235, 0.9);
        }}
        
        /* LaTeX 코드 박스 */
        .latex-box {{
            margin-bottom: 20px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .latex-box h4 {{
            margin: 0;
            padding: 10px 15px;
            background: #f3f4f6;
            font-size: 14px;
            font-weight: 600;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .latex-code {{
            padding: 15px;
            background: #e5e7eb;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 14px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            font-weight: bold;
            color: #111827;
        }}
        
        /* 렌더링 결과 */
        .rendered-math {{
            background: #2563eb;
            padding: 32px 20px;
            border-radius: 8px;
            text-align: center;
            min-height: 80px;
            position: relative;
            color: white;
            overflow: auto;
        }}
        
        .rendered-math h4 {{
            position: absolute;
            top: 10px;
            left: 15px;
            margin: 0;
            font-size: 14px;
            color: #bfdbfe;
        }}
        
        /* 줌 컨트롤 */
        .zoom-controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
            align-items: center;
        }}
        
        .zoom-btn {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .zoom-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}
        
        .zoom-level {{
            color: white;
            font-size: 12px;
            min-width: 40px;
            text-align: center;
        }}
        
        .rendered-math mjx-container {{
            color: white !important;
            max-width: 100%;
            overflow-x: auto;
        }}
        
        .rendered-math svg {{
            fill: white !important;
        }}
        
        .rendered-math svg * {{
            fill: white !important;
            stroke: white !important;
        }}
        
        /* 복사 버튼 */
        .copy-btn {{
            background: #2563eb;
            color: white;
            border: none;
            padding: 6px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            margin: 10px 15px;
        }}
        
        .copy-btn:hover {{
            background: #1d4ed8;
        }}
        
        /* 수정 표시 */
        .status-original {{
            color: #6b7280;
        }}
        
        .status-fixed {{
            color: #10b981;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <h1>SmartNougat LaTeX Processing Results</h1>
    
    <div class="stats">
        <h2>Processing Summary</h2>
        <p>Total Formulas: <strong>{total_formulas}</strong></p>
        <p>Fixed: <strong class="fixed">{fixed_count}</strong></p>
        <p>Fix Rate: <strong>{fixed_count/total_formulas*100:.1f}%</strong></p>
    </div>
'''

    # Add each formula
    images_dir = output_dir / "images"
    
    for result in formula_results:
        # Read image file and convert to base64
        img_path = images_dir / result['filename']
        img_base64 = ""
        if img_path.exists():
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Extract det_idx from filename for display
        det_num = int(result['filename'].split('_')[-1].replace('.png', ''))
        
        # Determine category
        category = result.get('category_id', '')
        category_class = 'category-inline' if category == 13 else 'category-block'
        category_name = 'Inline' if category == 13 else 'Block'
        
        # Status
        status_text = '🔧 Fixed' if result['was_fixed'] else '✓ Original'
        status_class = 'status-fixed' if result['was_fixed'] else 'status-original'
        
        html_content += f'''
    <div class="formula-card">
        <!-- Header -->
        <div class="card-header">
            <div>
                <span class="formula-number">#{det_num}</span> 
                Page {result['page_index'] + 1} | 
                <span class="{category_class}">{category_name} Formula</span>
            </div>
            <span class="{status_class}">{status_text}</span>
        </div>
        
        <!-- 1. 이미지 -->
        <div class="formula-image" id="image-{result['index']}">
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomOutImage({result['index']})">−</button>
                <span class="zoom-level" id="image-zoom-level-{result['index']}">100%</span>
                <button class="zoom-btn" onclick="zoomInImage({result['index']})">+</button>
            </div>
            <img src="data:image/png;base64,{img_base64}" alt="{result['filename']}" id="img-{result['index']}">
        </div>
        
        <!-- 2. 렌더링 결과 -->
        <div class="rendered-math" id="render-{result['index']}">
            <h4>렌더링 결과 / Rendered Result</h4>
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomOut({result['index']})">−</button>
                <span class="zoom-level" id="zoom-level-{result['index']}">100%</span>
                <button class="zoom-btn" onclick="zoomIn({result['index']})">+</button>
            </div>
            $${result['fixed_latex']}$$
        </div>
        
        <!-- 3. LaTeX 코드 (원본 + 수정본) -->
        <div class="latex-box">
            <!-- 원본 LaTeX -->
            <div class="latex-section">
                <h4>원본 LaTeX / Original LaTeX</h4>
                <div class="latex-code" id="original-{result['index']}">{html.escape(result['original_latex'])}</div>
                <button class="copy-btn" onclick="copyLatex('original-{result['index']}')">Copy Original</button>
            </div>
            
            <!-- 구분선 -->
            <div style="border-top: 2px solid #cbd5e1; margin: 20px 0;"></div>
            
            <!-- 수정된 LaTeX -->
            <div class="latex-section">
                <h4>수정된 LaTeX / Fixed LaTeX</h4>
                <div class="latex-code" id="fixed-{result['index']}">{html.escape(result['fixed_latex'])}</div>
                <button class="copy-btn" onclick="copyLatex('fixed-{result['index']}')">Copy Fixed</button>
            </div>
        </div>
    </div>
'''

    # Add JavaScript
    html_content += '''
    
    <script>
        // 줌 레벨 저장
        const zoomLevels = {};
        const imageZoomLevels = {};
        
        function zoomIn(id) {
            const current = zoomLevels[id] || 100;
            const newZoom = Math.min(current + 10, 300);
            zoomLevels[id] = newZoom;
            applyZoom(id, newZoom);
        }
        
        function zoomOut(id) {
            const current = zoomLevels[id] || 100;
            const newZoom = Math.max(current - 10, 50);
            zoomLevels[id] = newZoom;
            applyZoom(id, newZoom);
        }
        
        function applyZoom(id, zoom) {
            const element = document.querySelector(`#render-${id} mjx-container`);
            if (element) {
                element.style.transform = `scale(${zoom / 100})`;
                element.style.transformOrigin = 'center';
            }
            document.getElementById(`zoom-level-${id}`).textContent = zoom + '%';
        }
        
        // 이미지 줌 함수
        function zoomInImage(id) {
            const current = imageZoomLevels[id] || 100;
            const newZoom = Math.min(current + 10, 300);
            imageZoomLevels[id] = newZoom;
            applyImageZoom(id, newZoom);
        }
        
        function zoomOutImage(id) {
            const current = imageZoomLevels[id] || 100;
            const newZoom = Math.max(current - 10, 50);
            imageZoomLevels[id] = newZoom;
            applyImageZoom(id, newZoom);
        }
        
        function applyImageZoom(id, zoom) {
            const element = document.getElementById(`img-${id}`);
            if (element) {
                element.style.transform = `scale(${zoom / 100})`;
            }
            document.getElementById(`image-zoom-level-${id}`).textContent = zoom + '%';
        }
        
        function copyLatex(elementId) {
            const element = document.getElementById(elementId);
            const text = element.textContent;
            
            navigator.clipboard.writeText(text).then(() => {
                const button = element.nextElementSibling;
                const originalText = button.textContent;
                button.textContent = '✓ Copied';
                setTimeout(() => {
                    button.textContent = originalText;
                }, 2000);
            }).catch(err => {
                console.error('Copy failed:', err);
                alert('Copy failed');
            });
        }
        
        // Force white color for math
        function forceMathWhite() {
            document.querySelectorAll('.rendered-math mjx-container svg').forEach(svg => {
                svg.style.fill = 'white';
                svg.querySelectorAll('*').forEach(element => {
                    element.style.fill = 'white';
                    if (element.style.stroke && element.style.stroke !== 'none') {
                        element.style.stroke = 'white';
                    }
                });
            });
        }
        
        // Apply white color after MathJax renders
        if (window.MathJax) {
            window.MathJax.startup.promise.then(() => {
                forceMathWhite();
            });
        }
        
        // Also apply on load
        window.addEventListener('load', () => {
            setTimeout(forceMathWhite, 1000);
        });
        
        // Ctrl + 마우스 휠 줌
        document.addEventListener('wheel', (e) => {
            if (e.ctrlKey) {
                e.preventDefault();
                const mathElement = e.target.closest('.rendered-math');
                if (mathElement) {
                    const id = mathElement.id.replace('render-', '');
                    if (e.deltaY < 0) {
                        zoomIn(id);
                    } else {
                        zoomOut(id);
                    }
                }
            }
        });
    </script>
</body>
</html>'''

    # Write HTML file
    html_path = output_dir / "result_viewer_0715.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[Success] HTML viewer created: {html_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_fixed_viewer_simple.py <output_directory>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not create_fixed_viewer(output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()