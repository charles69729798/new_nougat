from docx2pdf import convert

# 간단한 변환
input_path = r"C:\test\1_AI.docx"
output_path = r"C:\test\1_AI.pdf"

print(f"Converting: {input_path} -> {output_path}")
convert(input_path, output_path)
print("Conversion complete!")