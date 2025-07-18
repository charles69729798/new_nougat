import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nougat-latex-ocr'))

# Set environment
os.environ['PYTHONPATH'] = ";".join(sys.path)

# Test imports
try:
    from smartnougat_gui_0717_v2 import SmartNougatGUI
    print("[OK] GUI module imported successfully")
    
    import tkinter as tk
    print("[OK] Tkinter available")
    
    # Test if SmartNougat can be imported
    from smartnougat_0715 import SmartNougatStandalone
    print("[OK] SmartNougat core module available")
    
    # Check for beautifulsoup4
    from bs4 import BeautifulSoup
    print("[OK] BeautifulSoup4 available")
    
    print("\nAll dependencies are OK. GUI should work.")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("\nPlease install missing dependencies:")
    print("pip install tkinter beautifulsoup4")