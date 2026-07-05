"""Import every library the project needs and print versions, so I know the
environment is good before doing anything else.

Run from the project root: python scripts/smoke_test.py
"""

import importlib
import sys

LIBS = [
    "mediapipe",
    "ultralytics",
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "pandas",
    "sklearn",
    "streamlit",
    "roboflow",
    "matplotlib",
    "tqdm",
]

missing = []

for name in LIBS:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<14} {getattr(mod, '__version__', '?')}")
    except ImportError as e:
        print(f"  {name:<14} FAILED ({e})")
        missing.append(name)

print()
if missing:
    print(f"missing {len(missing)}: {', '.join(missing)}")
    print("run: pip install -r requirements.txt")
    sys.exit(1)
print("everything imports, good to go")
