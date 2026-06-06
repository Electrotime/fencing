"""
Phase 0 smoke test — import every dependency and print its version.
Run from the project root: python scripts/smoke_test.py
"""

import importlib
import sys

LIBS = [
    ("mediapipe",    "mp"),
    ("ultralytics",  None),
    ("torch",        None),
    ("torchvision",  None),
    ("cv2",          None),
    ("numpy",        "np"),
    ("pandas",       "pd"),
    ("sklearn",      None),
    ("streamlit",    "st"),
    ("roboflow",     None),
    ("matplotlib",   None),
    ("tqdm",         None),
]

failed = []

for mod_name, alias in LIBS:
    try:
        mod = importlib.import_module(mod_name)
        version = getattr(mod, "__version__", "unknown")
        label = alias if alias else mod_name
        print(f"  {label:<14} {version}")
    except ImportError as e:
        print(f"  FAILED  {mod_name}: {e}")
        failed.append(mod_name)

print()
if failed:
    print(f"MISSING ({len(failed)}): {', '.join(failed)}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)
else:
    print("All imports OK — environment is ready.")
