#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX code fixer for common OCR errors
Fixes common LaTeX syntax errors from Nougat OCR output
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

class LaTeXFixer:
    """Fix common LaTeX OCR errors with enhanced pattern recognition"""
    
    # Common valid LaTeX commands (not exhaustive, but covers most frequent ones)
    COMMON_LATEX_COMMANDS = {
        # Greek letters
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega', 'Gamma', 'Delta',
        'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
        # Math operators
        'sum', 'prod', 'int', 'oint', 'bigcup', 'bigcap', 'lim', 'min', 'max',
        'sup', 'inf', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln', 'exp',
        # Font commands
        'mathrm', 'mathit', 'mathbf', 'mathsf', 'mathtt', 'mathcal', 'mathbb',
        'mathfrak', 'text', 'textrm', 'textit', 'textbf', 'textsf', 'texttt',
        # Size commands
        'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large',
        'Large', 'LARGE', 'huge', 'Huge',
        # Common commands
        'frac', 'sqrt', 'root', 'left', 'right', 'big', 'Big', 'bigg', 'Bigg',
        'cdot', 'cdots', 'ldots', 'vdots', 'ddots', 'times', 'div', 'pm', 'mp',
        'oplus', 'ominus', 'otimes', 'oslash', 'odot', 'circ', 'bullet',
        'overline', 'underline', 'overbrace', 'underbrace', 'hat', 'check',
        'tilde', 'acute', 'grave', 'dot', 'ddot', 'breve', 'bar', 'vec',
        # Relations
        'leq', 'geq', 'neq', 'approx', 'equiv', 'sim', 'simeq', 'propto',
        'subset', 'subseteq', 'supset', 'supseteq', 'in', 'ni', 'notin',
        # Arrows
        'rightarrow', 'leftarrow', 'leftrightarrow', 'Rightarrow', 'Leftarrow',
        'Leftrightarrow', 'uparrow', 'downarrow', 'updownarrow',
        # Delimiters
        'langle', 'rangle', 'lceil', 'rceil', 'lfloor', 'rfloor',
        # Other
        'quad', 'qquad', 'hspace', 'vspace', 'phantom', 'mathstrut',
        'begin', 'end', 'array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix',
        'cases', 'align', 'equation', 'label', 'ref', 'cite'
    }
    
    # Greek letters for pattern matching
    GREEK_LETTERS = [
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
        'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Pi', 'Rho', 'Sigma',
        'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'
    ]
    
    # Common OCR misrecognition patterns
    OCR_REPLACEMENTS = {
        'Nota': 'x+m',
        'tm': 'x+n',
        'xtm': 'x+m',
        'xtn': 'x+n',
        'Notm': 'x+m',
        'Notn': 'x+n'
    }
    
    def __init__(self):
        self.fix_count = 0
        self.total_fixes = {}
    
    def is_valid_command(self, cmd: str) -> bool:
        """Check if a LaTeX command is likely valid"""
        # Remove backslash and any arguments
        cmd_name = cmd.strip('\\').split('{')[0].split('[')[0].split('^')[0].split('_')[0]
        
        # Single letter commands are usually valid (like \n, \t, \r, etc.)
        if len(cmd_name) == 1 and cmd_name.isalpha():
            return True
            
        # Check against known commands
        return cmd_name in self.COMMON_LATEX_COMMANDS
    
    def fix_latex_code(self, latex: str) -> Tuple[str, List[str]]:
        """
        Fix common LaTeX errors and return fixed code with list of fixes applied
        """
        if not latex:
            return latex, []
        
        original = latex
        fixes_applied = []
        
        # Phase 1: Fix structural issues first
        latex, structural_fixes = self.fix_structural_issues(latex)
        fixes_applied.extend(structural_fixes)
        
        # Phase 2: Fix OCR-specific misrecognitions
        latex, ocr_fixes = self.fix_ocr_patterns(latex)
        fixes_applied.extend(ocr_fixes)
        
        # Phase 3: Clean up formatting
        latex, format_fixes = self.fix_formatting_issues(latex)
        fixes_applied.extend(format_fixes)
        
        if latex != original:
            self.fix_count += 1
            for fix in fixes_applied:
                self.total_fixes[fix] = self.total_fixes.get(fix, 0) + 1
        
        return latex, fixes_applied
    
    def fix_structural_issues(self, latex: str) -> Tuple[str, List[str]]:
        """Fix structural LaTeX issues"""
        fixes = []
        original = latex
        
        # 1. Fix excessive braces like {{{n}}} -> {n}
        if '{{{' in latex:
            latex = re.sub(r'\{\{\{([^}]+)\}\}\}', r'{\1}', latex)
            latex = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', latex)
            fixes.append("Fixed excessive braces")
        
        # 2. Fix unclosed \mathrm{ environments
        before_close = latex
        latex = self.close_mathrm_properly(latex)
        if before_close != latex:
            fixes.append("Fixed unclosed \\mathrm{}")
        
        # 3. Attempt to detect fraction structure
        # Pattern: P로 시작하는 복잡한 수식이 = 앞에 있는 경우
        if '=' in latex and len(latex) > 100:  # 긴 수식일 때만
            eq_pos = latex.find('=')
            before_eq = latex[:eq_pos].strip()
            after_eq = latex[eq_pos+1:].strip()
            
            # 휴리스틱: P로 시작하고 긴 수식이면 분수일 가능성
            if re.match(r'^[a-zA-Z]+[\']*[\^_]', before_eq) and len(before_eq) > 50:
                # 분자와 분모를 구분하는 패턴 찾기
                # 패턴 1: 연속된 공백이나 줄바꿈
                parts = re.split(r'\s{3,}|\n', before_eq, maxsplit=1)
                if len(parts) == 2:
                    numerator = parts[0].strip()
                    denominator = parts[1].strip()
                    latex = f"\\frac{{{numerator}}}{{{denominator}}} = {after_eq}"
                    fixes.append("Detected and fixed fraction structure")
                
                # 패턴 2: 특정 구분자가 있는 경우 (예: 긴 공백 후 숫자로 시작)
                elif re.search(r'(.+?)\s+(\d+\s*\·)', before_eq):
                    match = re.search(r'(.+?)\s+(\d+\s*\·.+)', before_eq)
                    if match:
                        numerator = match.group(1).strip()
                        denominator = match.group(2).strip()
                        latex = f"\\frac{{{numerator}}}{{{denominator}}} = {after_eq}"
                        fixes.append("Detected fraction structure by pattern")
        
        # 4. 등호 좌변이 분자에 중복되는 문제 수정
        if '=' in latex and r'\frac{' in latex:
            # 더 유연한 패턴 매칭
            eq_pos = latex.find('=')
            if eq_pos > 0:
                left_side = latex[:eq_pos].strip()
                frac_start = latex.find(r'\frac{', eq_pos)
                
                if frac_start > eq_pos:
                    # 분자 시작 부분 확인
                    numerator_start = frac_start + 6  # \frac{ 길이
                    # 좌변과 비슷한 패턴이 분자 시작에 있는지 확인
                    if latex[numerator_start:].startswith(left_side):
                        # 중복 제거
                        latex = latex[:numerator_start] + latex[numerator_start + len(left_side):]
                        fixes.append("Fixed equation duplication in fraction")
                    # 변형된 형태 체크 (예: _mP' → _m{P')
                    elif left_side.replace("'", "") in latex[numerator_start:numerator_start+len(left_side)+5]:
                        # 더 복잡한 패턴 처리
                        escaped_side = re.escape(left_side).replace('_', '_[^{]*{?').replace('^', r'\^')
                        pattern = f"\\\\frac{{{escaped_side}"
                        latex = re.sub(pattern, r'\\frac{', latex)
                        fixes.append("Fixed complex equation duplication")
        
        # 5. 분수 내부의 \mathrm{} 제거
        if r'\frac{' in latex and r'\mathrm{' in latex:
            # 분자 내부의 \mathrm{} 제거
            latex = re.sub(r'\\frac\{\\mathrm\{([^}]+)\}', r'\\frac{\1', latex)
            # 분모 내부의 \mathrm{} 제거
            latex = re.sub(r'\}\{\\mathrm\{([^}]+)\}\}', r'}{\1}', latex)
            if 'mathrm' not in latex or latex.count('mathrm') < original.count('mathrm'):
                fixes.append("Removed \\mathrm{} inside fraction")
        
        # 6. array 환경의 중첩 분수 감지
        if r'\begin{array}{c}' in latex:
            latex, array_fixes = self.detect_nested_fractions(latex)
            fixes.extend(array_fixes)
        
        return latex, fixes
    
    def detect_nested_fractions(self, latex: str) -> Tuple[str, List[str]]:
        """array 환경에서 중첩된 분수 구조를 감지하고 변환"""
        fixes = []
        
        # array 환경 감지
        array_match = re.search(r'\\begin\{array\}\{c\}(.+?)\\end\{array\}', latex, re.DOTALL)
        if array_match:
            array_content = array_match.group(1)
            
            # \\ 로 줄 분리
            lines = [line.strip() for line in array_content.split('\\\\') if line.strip()]
            
            if len(lines) >= 2:
                # 각 줄의 복잡도 계산
                line_scores = []
                for i, line in enumerate(lines):
                    score = self.calculate_line_complexity(line)
                    line_scores.append((i, score, line))
                
                # 가장 복잡한 줄을 분수선으로 가정
                line_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 연속된 줄들 중 가운데를 찾기
                if len(lines) == 2:
                    # 단순 분수
                    result = f"\\frac{{{lines[0]}}}{{{lines[1]}}}"
                    latex = latex.replace(array_match.group(0), result)
                    fixes.append("Converted array to fraction")
                elif len(lines) >= 3:
                    # 복잡한 구조 - 휴리스틱 적용
                    # 중간 지점을 분수선으로
                    mid = len(lines) // 2
                    numerator = ' '.join(lines[:mid])
                    denominator = ' '.join(lines[mid:])
                    result = f"\\frac{{{numerator}}}{{{denominator}}}"
                    latex = latex.replace(array_match.group(0), result)
                    fixes.append("Converted complex array to nested fraction")
        
        return latex, fixes
    
    def calculate_line_complexity(self, line: str) -> int:
        """줄의 복잡도 점수 계산"""
        score = 0
        
        # 길이
        score += len(line)
        
        # 수학 기호 개수
        math_symbols = [r'\cdot', r'\{', r'\}', r'\left', r'\right', '+', '-', '=', r'\frac']
        for symbol in math_symbols:
            score += line.count(symbol) * 2
        
        # 괄호 깊이
        brace_depth = 0
        max_depth = 0
        for char in line:
            if char == '{':
                brace_depth += 1
                max_depth = max(max_depth, brace_depth)
            elif char == '}':
                brace_depth -= 1
        score += max_depth * 10
        
        # 특정 패턴 보너스
        if '1 -' in line or '(1 -' in line:
            score += 20
        if r'\cdot' in line and r'\frac' not in line:
            score += 15
        
        return score
    
    def fix_ocr_patterns(self, latex: str) -> Tuple[str, List[str]]:
        """Fix OCR-specific misrecognition patterns"""
        fixes = []
        original = latex
        
        # 4. 불필요한 전체 \mathrm{} 제거 (먼저 처리)
        # 전체를 감싸는 경우만 제거
        if latex.startswith(r'\mathrm{') and latex.endswith('}'):
            # 내부 중괄호 개수 확인
            inner = latex[8:-1]  # \mathrm{ 와 마지막 } 제거
            brace_count = 0
            valid = True
            for char in inner:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                if brace_count < 0:
                    valid = False
                    break
            if valid and brace_count == 0:
                latex = inner
                fixes.append("Removed unnecessary outer \\mathrm{}")
        
        # 1. Fix left subscript pattern for ALL characters
        # Pattern: \g|^{\cal{X}} → _mX' where X is any letter or Greek letter
        # Build pattern for all possible characters
        all_chars = '|'.join(['[A-Za-z]'] + ['\\\\' + g for g in self.GREEK_LETTERS])
        left_subscript_pattern = rf'\\g\|\\?\^?\{{\\cal\{{({all_chars})\}}\}}'
        if re.search(left_subscript_pattern, latex):
            latex = re.sub(left_subscript_pattern, r"_m\1'", latex)
            fixes.append("Fixed left subscript misrecognition")
        
        # 2. Fix \cal{X} that should be X' for ALL characters
        cal_prime_pattern = rf'\\cal\{{({all_chars})\\?\}}\s*\^'
        if re.search(cal_prime_pattern, latex):
            latex = re.sub(cal_prime_pattern, r"\1'^", latex)
            fixes.append("Fixed prime misrecognized as \\cal{}")
        
        # Also fix standalone \cal{X} → X'
        latex = re.sub(rf'\\cal\{{({all_chars})\}}(?!\^)', r"\1'", latex)
        
        # 3. Fix \mathrm{^{\prime}} → ' for any context
        latex = re.sub(r'\\mathrm\{\\?\^?\{\\prime\}\}', "'", latex)
        latex = re.sub(r'\\?\^?\{\\prime\}', "'", latex)
        if "prime" in original and "prime" not in latex:
            fixes.append("Fixed prime notation")
        
        # 4. Fix semicolon that should be colon in equations
        colon_pattern = r'([a-zA-Z0-9\)])\s*;\s*([a-zA-Z0-9\(\\])'
        if re.search(colon_pattern, latex):
            latex = re.sub(colon_pattern, r'\1:\2', latex)
            fixes.append("Fixed semicolon to colon")
        
        # 5. Fix \g| patterns more intelligently
        if r'\g|' in latex:
            # Try to detect context - if followed by ^, it's likely a left subscript
            latex = re.sub(r'\\g\|\\?\^', '_m', latex)
            latex = re.sub(r'\\g\|', '', latex)  # Remove any remaining
            fixes.append("Fixed \\g| pattern")
        
        # 6. Fix \cal{} to \mathcal{} for proper rendering
        if r'\cal{' in latex:
            latex = re.sub(r'\\cal\{([^}]+)\}', r'\\mathcal{\1}', latex)
            fixes.append("Fixed \\cal{} to \\mathcal{}")
        
        # 7. Fix scriptsize in subscripts - more intelligent replacement
        scriptsize_pattern = r'_\{\\scriptsize\{([^}]+)\}\}'
        if re.search(scriptsize_pattern, latex):
            def fix_scriptsize(match):
                content = match.group(1)
                # Check OCR replacements
                for old, new in self.OCR_REPLACEMENTS.items():
                    if old in content:
                        return f'_{{{new}}}'
                # If no specific replacement, just remove scriptsize
                return f'_{{{content}}}'
            
            latex = re.sub(scriptsize_pattern, fix_scriptsize, latex)
            fixes.append("Fixed subscript with \\scriptsize")
        
        # 8. Fix \mathrm{\scriptsize{}} issues
        latex = re.sub(r'\\mathrm\{\\scriptsize\{([^}]*)\}\}', r'\\scriptsize{\1}', latex)
        if r'\mathrm{\scriptsize{' in original and r'\mathrm{\scriptsize{' not in latex:
            fixes.append("Fixed \\mathrm{\\scriptsize{}} usage")
        
        # 9. 왼쪽 아래첨자 다음 문자 대문자 변환
        # Pattern: _소문자 다음의 소문자를 대문자로
        left_subscript_cap = r'_([a-z])([a-z])'
        if re.search(left_subscript_cap, latex):
            latex = re.sub(left_subscript_cap, lambda m: f'_{m.group(1)}{m.group(2).upper()}', latex)
            fixes.append("Fixed left subscript capitalization")
        
        # 10. 그리스 문자 복구 (컨텍스트 기반)
        greek_replacements = [
            # a → α
            (r'\ba_1\b', r'\\alpha_1'),
            (r'\ba_2\b', r'\\alpha_2'), 
            (r'\ba_{1}', r'\\alpha_{1}'),
            (r'\ba_{2}', r'\\alpha_{2}'),
            (r'\ba_\{1\}', r'\\alpha_{1}'),
            (r'\ba_\{2\}', r'\\alpha_{2}'),
            # 독립된 a (변수명이 아닌 경우)
            (r'(?<![a-zA-Z])a(?=\s*[\+\-\*\=/])', r'\\alpha'),
            # b → β
            (r'(?<![a-zA-Z])b(?![a-zA-Z0-9_])', r'\\beta'),
            (r"b'", r"\\beta'"),  # b' → β'
            # r → γ
            (r'(?<![a-zA-Z])r(?![a-zA-Z0-9_])', r'\\gamma'),
            # 1 → l (대문자 1이 소문자 l로 오인식)
            (r'(?<![0-9])1(?=[a-zA-Z])', r'l'),
            # \mathrm{'} → \beta' (mathrm 안의 프라임)
            (r"\\mathrm\{'?\}", r"\\beta'"),
        ]
        
        for pattern, replacement in greek_replacements:
            if re.search(pattern, latex):
                latex = re.sub(pattern, replacement, latex)
                fixes.append(f"Fixed Greek letter")
        
        # 11. 대문자 다음 아래첨자 패턴 수정
        # N_{\scriptsize{...}} 패턴 개선
        cap_subscript = r'([A-Z])_\{\\scriptsize\{([^}]+)\}\}'
        def fix_cap_subscript(match):
            letter = match.group(1)
            content = match.group(2)
            
            # 특정 패턴 교체
            subscript_fixes = {
                'Nota': 'x+m',
                'tm': 'x+n',
                'xtm': 'x+m', 
                'xtn': 'x+n',
                'Notm': 'x+m',
                'Notn': 'x+n',
            }
            
            for old, new in subscript_fixes.items():
                if old == content:
                    return f'{letter}_{{{new}}}'
            
            # 일반 패턴: 2-3글자면 변수+연산자로 분리
            if len(content) == 2 and content.isalpha():
                return f'{letter}_{{{content[0]}+{content[1]}}}'
            
            return f'{letter}_{{{content}}}'  # 기본: scriptsize만 제거
        
        if re.search(cap_subscript, latex):
            latex = re.sub(cap_subscript, fix_cap_subscript, latex)
            fixes.append("Fixed capital letter subscripts")
        
        # 12. 소문자 다음의 overline 문자들을 직각 모양으로 변환
        if '─' in latex or '┐' in latex:
            # 소문자 + ─ 또는 ┐ → \overset{\urcorner}{소문자}
            latex = re.sub(r'([a-z])\s*[─┐]', r'\\overset{\\urcorner}{\\1}', latex)
            fixes.append("Fixed corner overline notation (─/┐ → \\overset{\\urcorner}{})")
        
        # 12-1. 기존 \overline{n}을 직각 모양으로 변환
        if r'\overline{n}' in latex:
            latex = re.sub(r'\\overline\{n\}', r'\\overset{\\urcorner}{n}', latex)
            fixes.append("Changed \\overline{n} to corner notation")
        
        # 13. 다른 overline 유사 문자들도 직각 모양으로 처리
        overline_chars = ['━', '¯', '‾', '￣', '⎯']  # 가능한 overline 문자들
        for char in overline_chars:
            if char in latex:
                # 소문자 + overline 특수문자 → \overset{\urcorner}{소문자}
                pattern = rf'([a-z])\s*{re.escape(char)}'
                if re.search(pattern, latex):
                    latex = re.sub(pattern, r'\\overset{\\urcorner}{\\1}', latex)
                    fixes.append(f"Fixed corner overline notation ({char} → \\overset{{\\urcorner}}{{}})")
        
        # 14. alpha 아래첨자 수정
        # \alpha_{1}{1} → \alpha_1
        if r'\alpha_{' in latex:
            latex = re.sub(r'\\alpha_\{(\d)\}\{\\1\}', r'\\alpha_\1', latex)
            latex = re.sub(r'\\alpha_\{1\}\{2\}', r'\\alpha_2', latex)
            fixes.append("Fixed alpha subscript notation")
        
        # 15. 문자 인식 오류 수정
        character_replacements = [
            # A/B를 *로 인식
            (r'\^\{\\ast\}', '^A'),
            (r'\^\{\\ast1\}', '^{A1}'),
            (r'_\{\\ast\\ast\}', '_{x+k}'),
            
            # 그리스 문자 오인식
            (r'\\mathrm\{r\}\^\{', 'Ψ^{'),
            (r'LT/\\nu', 'LTC'),  # 한글 제거
            
            # M/N 혼동 (특정 패턴)
            (r'\\mathbb\{N\}_\{x\}', 'M_x'),
            (r'\\mathbb\{N\}\^\{', 'M^{'),
            (r'\\mathbb\{C\}', 'M'),  # C로 잘못 인식된 M
        ]
        
        for pattern, replacement in character_replacements:
            if re.search(pattern, latex):
                latex = re.sub(pattern, replacement, latex)
                fixes.append("Fixed character recognition error")
        
        # 16. 아래첨자 + 기호 복원
        subscript_plus_patterns = [
            (r'_\{x\s*n\}', '_{x+n}'),
            (r'_\{x\s*m\}', '_{x+m}'),
            (r'_\{xnk\}', '_{x+k}'),
            (r'_\{x\s*n\s*k\}', '_{x+k}'),
            (r':n(?=[)\s\}])', '+n'),
            (r':20(?=[)\s\}])', '+20'),
            (r'_\{\\kappa', '_{x'),  # κ를 x로
            (r'\\kappa', 'x'),  # 독립된 κ도 x로
        ]
        
        for pattern, replacement in subscript_plus_patterns:
            if re.search(pattern, latex):
                latex = re.sub(pattern, replacement, latex)
                fixes.append("Fixed subscript plus notation")
        
        # 17. 과도한 반복 패턴 제거
        # (1-q)가 5번 이상 반복되면 축약
        repeat_pattern = r'(\(1\s*-\s*[^)]+\)\s*\\cdot\s*){5,}'
        if re.search(repeat_pattern, latex):
            latex = re.sub(repeat_pattern, r'(1-q)^n \\cdot ', latex)
            fixes.append("Fixed excessive repetition")
        
        # 18. 특수 기호 정리
        special_symbols = [
            (r'\\mathrm\{\\boldmath\{~1~\}\}', 'M̄'),  # 1→M̄
            (r'\\boldmath\{~1~\}', 'M̄'),
            (r'\\mathrm\{\\boldmath\{g\}\}\^\\ast', "β'"),
            (r'\\boldmath\{g\}\^\\ast', "β'"),
            (r'\\mathrm\{\\boldmath\{a\}\}', 'α'),
            (r'\\boldmath\{a\}', 'α'),
            (r'D\\!E', 'DE'),  # 불필요한 공백 제거
        ]
        
        for pattern, replacement in special_symbols:
            if re.search(pattern, latex):
                latex = re.sub(pattern, replacement, latex)
                fixes.append("Fixed special symbol notation")
        
        # 19. \mathrm{\boldmath~\scriptstyle ...} 잘못된 중첩 수정
        # Page 3 #0 패턴
        if r'\mathrm{\boldmath~\scriptstyle' in latex:
            # \mathrm{\boldmath~\scriptstyle k~} → k
            latex = re.sub(r'\\mathrm\{\\boldmath~\\scriptstyle\s*([^}~]+)~*\}', r'\1', latex)
            fixes.append("Fixed \\mathrm{\\boldmath~\\scriptstyle} nesting")
        
        # 20. 과도한 틸드(~) 제거
        # 연속된 틸드 제거
        if '~~' in latex:
            latex = re.sub(r'~{2,}', '~', latex)
            fixes.append("Removed excessive tildes")
        
        # 단어 끝의 불필요한 틸드 제거
        latex = re.sub(r'([a-zA-Z0-9])~(?=[\s\}\)])', r'\1', latex)
        
        # 21. 과도하게 중첩된 아래첨자 단순화
        # Page 4 #0, #1 패턴: _{{\bf\Lambda}_{{\bf\Lambda}_{{\bf\Lambda}_{{\bf\Lambda}}}}}
        nested_subscript_pattern = r'_\{\{[^}]*\}_\{\{[^}]*\}_\{\{[^}]*\}_\{\{[^}]*\}\}\}\}\}'
        if re.search(nested_subscript_pattern, latex):
            # 가장 깊은 중첩부터 단순화
            latex = re.sub(r'_\{\{\\bf\s*([^}]+)\}_\{\{\\bf\s*([^}]+)\}_\{\{\\bf\s*([^}]+)\}_\{\{\\bf\s*([^}]+)\}\}\}\}\}', 
                          r'_{\1_{\2_{\3_{\4}}}}', latex)
            fixes.append("Simplified deeply nested subscripts")
        
        # 22. 구식 명령어 현대화
        # \bf → \mathbf, \mit → \mathit
        if r'\bf' in latex or r'\mit' in latex:
            # 단독 \bf, \mit (중괄호 없이)
            latex = re.sub(r'\\bf(?![a-zA-Z])', r'\\mathbf', latex)
            latex = re.sub(r'\\mit(?![a-zA-Z])', r'\\mathit', latex)
            # 중괄호 내부의 \bf, \mit
            latex = re.sub(r'\{\\bf\s+', r'{\\mathbf{', latex)
            latex = re.sub(r'\{\\mit\s+', r'{\\mathit{', latex)
            fixes.append("Modernized obsolete commands")
        
        # 23. 잘못된 명령어 조합 수정
        # \mathrm 내부의 다른 스타일 명령어 제거
        if r'\mathrm{' in latex:
            # \mathrm 내부의 \boldmath, \scriptstyle 등 제거
            def clean_mathrm_content(match):
                content = match.group(1)
                # 스타일 명령어 제거
                content = re.sub(r'\\(boldmath|scriptstyle|scriptscriptstyle|textstyle|displaystyle)\s*', '', content)
                # 불필요한 틸드 제거
                content = content.replace('~', ' ').strip()
                return f'\\mathrm{{{content}}}'
            
            latex = re.sub(r'\\mathrm\{([^}]+)\}', clean_mathrm_content, latex)
            fixes.append("Cleaned \\mathrm{} content")
        
        # 24. \Lambda와 같은 대문자 그리스 문자 앞의 \bf 제거
        greek_caps = ['Lambda', 'Gamma', 'Delta', 'Theta', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega']
        for letter in greek_caps:
            pattern = rf'\{{\\bf\s*\\{letter}\}}'
            if re.search(pattern, latex):
                latex = re.sub(pattern, rf'\\{letter}', latex)
                fixes.append(f"Removed \\bf before \\{letter}")
        
        # 25. 중괄호 균형 맞추기 (마지막 단계)
        brace_count = latex.count('{') - latex.count('}')
        if brace_count > 0:
            latex += '}' * brace_count
            fixes.append("Added missing closing braces")
        elif brace_count < 0:
            # 닫는 중괄호가 더 많은 경우 - 끝에서부터 제거
            latex = latex.rstrip('}')
            latex += '}' * (latex.count('{') - latex.count('}'))
            fixes.append("Balanced excess closing braces")
        
        return latex, fixes
    
    def fix_formatting_issues(self, latex: str) -> Tuple[str, List[str]]:
        """Fix formatting and spacing issues"""
        fixes = []
        
        # 1. Fix excessive spacing
        if r'\,' in latex:
            # 3개 이상 연속된 \, 제거
            latex = re.sub(r'(\\,\s*){3,}', r'\\,', latex)
            # 2개 연속된 \, 제거
            latex = re.sub(r'(\\,\s*){2}', r'\\,', latex)
            # 공백과 \, 조합 정리
            latex = re.sub(r'\s*\\,\s*\\,\s*', r'\\,', latex)
            fixes.append("Fixed excessive spacing")
        
        # 2. Fix subscript spacing
        latex = re.sub(r'([A-Za-z])\s*\\,\s*_', r'\1_', latex)
        if r'\,_' in latex:
            fixes.append("Fixed subscript spacing")
        
        # 3. Remove empty super/subscripts
        latex = re.sub(r'\^\{\s*\}', '', latex)
        latex = re.sub(r'_\{\s*\}', '', latex)
        
        # 4. Fix spacing around operators
        latex = re.sub(r'\\,\s*\\cdot\s*\\,', r' \\cdot ', latex)
        
        # 5. 원 안의 숫자 제거
        circled_nums = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', 
                        '⑪', '⑫', '⑬', '⑭', '⑮', '⑯', '⑰', '⑱', '⑲', '⑳']
        for num in circled_nums:
            if num in latex:
                latex = latex.replace(num, '')
                fixes.append("Removed circled numbers")
                break  # 한 번만 기록
        
        # 6. 단독 괄호 숫자 제거 
        # 수식 끝이나 공백 뒤에 오는 (1), (2) 등
        if re.search(r'\s+\(\d+\)', latex) or re.search(r'\(\d+\)\s*$', latex):
            latex = re.sub(r'\s+\(\d+\)(?=\s|$|\\)', ' ', latex)  # 공백 뒤
            latex = re.sub(r'\s*\(\d+\)\s*$', '', latex)  # 문장 끝
            fixes.append("Removed standalone number in parentheses")
        
        return latex, fixes
    
    def close_mathrm_properly(self, latex: str) -> str:
        """More intelligent \mathrm closing"""
        result = []
        i = 0
        while i < len(latex):
            if latex[i:i+8] == r'\mathrm{':
                # Found \mathrm{, now track braces
                result.append(r'\mathrm{')
                i += 8
                brace_count = 1
                
                while i < len(latex) and brace_count > 0:
                    if latex[i] == '{':
                        brace_count += 1
                    elif latex[i] == '}':
                        brace_count -= 1
                    result.append(latex[i])
                    i += 1
                
                # If unclosed, add closing braces
                if brace_count > 0:
                    result.extend('}' * brace_count)
            else:
                result.append(latex[i])
                i += 1
        
        return ''.join(result)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_latex.py <results.json>")
        print("Example: python fix_latex.py results.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    # Create output filename
    output_file = input_file.parent / f"{input_file.stem}_fixed.json"
    
    print(f"[Reading] {input_file}")
    
    # Read JSON
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)
    
    # Initialize fixer
    fixer = LaTeXFixer()
    
    # Process each entry
    print(f"[Processing] {len(data)} entries...")
    
    for item in data:
        if 'latex' in item and item['latex']:
            original_latex = item['latex']
            fixed_latex, fixes = fixer.fix_latex_code(original_latex)
            
            item['latex'] = fixed_latex
            item['latex_fixes'] = fixes
            item['latex_original'] = original_latex if fixes else None
            
            if fixes:
                print(f"  [Fixed] #{item['index'] + 1} {item['filename']}: {', '.join(fixes)}")
    
    # Save fixed JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Total entries: {len(data)}")
    print(f"  Entries fixed: {fixer.fix_count}")
    
    if fixer.total_fixes:
        print(f"\n[Fixes applied]")
        for fix_type, count in sorted(fixer.total_fixes.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {fix_type}: {count}")
    
    print(f"\n[Completed] Fixed JSON saved to: {output_file}")
    
    return str(output_file)

if __name__ == "__main__":
    main()