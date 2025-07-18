import urllib.request
import os

print("Downloading MathJax for offline use...")

# Create resources directory
resources_dir = r"C:\SmartNougat\resources\mathjax\es5"
os.makedirs(resources_dir, exist_ok=True)

# Download MathJax
cdn_url = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
output_path = os.path.join(resources_dir, "tex-svg.js")

try:
    print(f"Downloading from: {cdn_url}")
    with urllib.request.urlopen(cdn_url) as response:
        mathjax_content = response.read()
    
    with open(output_path, 'wb') as f:
        f.write(mathjax_content)
    
    print(f"Success! MathJax saved to: {output_path}")
    print(f"File size: {len(mathjax_content):,} bytes")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nAlternative: Download manually from:")
    print(cdn_url)
    print(f"And save to: {output_path}")