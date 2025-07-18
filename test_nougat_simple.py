import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer
import numpy as np
import os

print("=== Nougat 간단 테스트 ===\n")

# 모델 로드
model_path = r"C:\SmartNougat\new_nougat-main\models"
print(f"모델 경로: {model_path}")

try:
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("[OK] 모델 로드 성공")
except Exception as e:
    print(f"[FAIL] 모델 로드 실패: {e}")
    exit(1)

# 테스트 이미지
img_path = r"C:\SmartNougat\new_nougat-main\src\output\1_AI_smartnougat_20250718_193558\images\formula_page0_002.png"

if os.path.exists(img_path):
    print(f"\n테스트 이미지: {img_path}")
    
    # 이미지 로드
    img = Image.open(img_path).convert('RGB')
    print(f"이미지 크기: {img.size}")
    
    # 간단한 전처리 (processor 없이)
    import torchvision.transforms as transforms
    
    # Nougat 모델에 맞는 전처리
    transform = transforms.Compose([
        transforms.Resize((896, 672)),  # Nougat 기본 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pixel_values = transform(img).unsqueeze(0)
    print(f"Tensor shape: {pixel_values.shape}")
    
    # 생성
    task_prompt = tokenizer.bos_token
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    
    print("\nLaTeX 생성 중...")
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
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
    sequence = sequence.replace(tokenizer.eos_token, "").replace(
        tokenizer.pad_token, "").replace(tokenizer.bos_token, "").strip()
    
    print(f"\n생성된 LaTeX:\n{sequence}")
    
else:
    print(f"\n[FAIL] 이미지 파일을 찾을 수 없습니다: {img_path}")

print("\n=== 테스트 완료 ===")