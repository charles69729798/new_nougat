# SmartNougat - PDF/DOCX to LaTeX Converter

🚀 **완전히 작동하는** PDF/DOCX 문서의 수식을 LaTeX로 변환하는 도구입니다.

## 🎯 주요 기능
- **YOLO v8 수식 감지**: 문서에서 수식을 자동으로 감지
- **Nougat LaTeX 변환**: 감지된 수식을 정확한 LaTeX 코드로 변환  
- **독립 실행형 HTML**: 모든 이미지와 MathJax가 내장된 완전한 HTML 파일 생성
- **GUI & CLI**: 사용하기 쉬운 그래픽 인터페이스와 명령줄 도구
- **완전한 오프라인 작동**: 인터넷 연결 없이도 모든 기능 사용 가능

## Requirements
- Windows 10/11 (64-bit)
- 8GB RAM minimum
- 5GB free disk space
- Internet connection for first-time setup

## Installation

### Step 1: Download and Install
```batch
# Download from GitHub
https://github.com/charles69729798/new_nougat/archive/refs/heads/main.zip

# Extract and run
install.bat
```

### Step 2: Model Setup

#### Option A: Automatic Download (if internet works)
```batch
download_models.bat
```

#### Option B: Manual Download (if automatic fails)
1. Download all files from: https://huggingface.co/Norm/nougat-latex-base/tree/main
2. Save to `models` folder
3. Run:
```batch
setup_downloaded_models.bat
```

### Step 3: Run SmartNougat
- Desktop shortcut: **SmartNougat**
- Or run: `run_smartnougat.bat`

## File Structure
```
new_nougat/
├── install.bat                    # Main installer
├── download_models.bat            # Model downloader
├── setup_downloaded_models.bat    # Offline model setup
├── run_smartnougat.bat           # Run SmartNougat
├── src/                          # Source code
├── scripts/                      # Helper scripts
└── models/                       # AI models (created during setup)
```

## Troubleshooting

### Installation Issues
- Run as Administrator
- Check internet connection
- Ensure 5GB free space

### Model Download Issues
If automatic download fails:
1. Manually download from Hugging Face
2. Place files in `models` folder
3. Run `setup_downloaded_models.bat`

### Runtime Issues
- Delete `venv` folder and reinstall
- Check if all model files are present
- Ensure enough RAM (4GB+ free)

## Output Location
```
%USERPROFILE%\Documents\Nougat_result\
```

## License
Uses open-source components. See individual licenses.