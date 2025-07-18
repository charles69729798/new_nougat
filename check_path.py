import sys
print("Python paths:")
for p in sys.path:
    print(f"  {p}")

print("\nChecking nougat_latex import:")
try:
    import nougat_latex
    print("✓ nougat_latex imported successfully")
    print(f"  Location: {nougat_latex.__file__}")
except ImportError as e:
    print(f"✗ Failed to import nougat_latex: {e}")

print("\nChecking NougatLaTexProcessor:")
try:
    from nougat_latex import NougatLaTexProcessor
    print("✓ NougatLaTexProcessor imported successfully")
except ImportError as e:
    print(f"✗ Failed to import NougatLaTexProcessor: {e}")

print("\nChecking util:")
try:
    from nougat_latex.util import process_raw_latex_code
    print("✓ process_raw_latex_code imported successfully")
except ImportError as e:
    print(f"✗ Failed to import process_raw_latex_code: {e}")