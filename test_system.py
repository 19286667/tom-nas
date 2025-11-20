#!/usr/bin/env python
"""ToM-NAS System Test"""
import os
print("="*60)
print("ToM-NAS System Check")
print("="*60)
dirs = ["src/core", "src/agents", "src/world"]
for d in dirs:
    status = "✓" if os.path.exists(d) else "✗"
    print(f"  {status} {d}")
print("="*60)
print("Structure ready. Add module files to src/ directories.")
