"""Setup models for offline use"""
import os
import sys
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent.parent
models_dir = script_dir / "models"
cache_dir = models_dir / ".cache"

# Create directories
models_dir.mkdir(exist_ok=True)
cache_dir.mkdir(exist_ok=True)

# Set environment variables
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_HOME'] = str(cache_dir)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)

print("Setting up models...")
print(f"Models directory: {models_dir}")
print(f"Cache directory: {cache_dir}")

try:
    from huggingface_hub import snapshot_download
    from transformers import VisionEncoderDecoderModel, NougatProcessor
    
    print("\nDownloading Nougat LaTeX model...")
    print("This is a one-time download (~1.5GB)")
    print("Please wait...")
    
    # Download model
    model_path = snapshot_download(
        repo_id="Norm/nougat-latex-base",
        cache_dir=cache_dir,
        local_files_only=False,
        resume_download=True
    )
    
    print(f"\nModel downloaded to: {model_path}")
    
    # Try to load model to verify
    print("\nVerifying model...")
    processor = NougatProcessor.from_pretrained("Norm/nougat-latex-base", cache_dir=cache_dir)
    model = VisionEncoderDecoderModel.from_pretrained("Norm/nougat-latex-base", cache_dir=cache_dir)
    
    print("✓ Model verified successfully!")
    print("\n✓ Model setup complete!")
    
except ImportError as e:
    print(f"\nERROR: Required packages not installed: {e}")
    print("Please ensure transformers is installed correctly")
    
except Exception as e:
    print(f"\nERROR during model download: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Try running again (download will resume)")
    print("3. If using proxy, set HTTP_PROXY and HTTPS_PROXY")
    print("\nModel will be downloaded on first use if not available")