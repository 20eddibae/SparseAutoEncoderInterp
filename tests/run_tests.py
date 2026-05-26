#!/usr/bin/env python3
"""Minimal test runner (the conda env has no pytest). Imports every test_*.py
and calls each test_* function; exits non-zero on any failure."""
from __future__ import annotations
import importlib, os, sys, types, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)


def main() -> int:
    mods = [f[:-3] for f in sorted(os.listdir(HERE))
            if f.startswith("test_") and f.endswith(".py")]
    ran = fails = 0
    for m in mods:
        try:
            mod = importlib.import_module(m)
        except Exception as e:
            fails += 1
            print(f"IMPORT FAIL {m}: {e!r}")
            continue
        for name in dir(mod):
            fn = getattr(mod, name)
            if name.startswith("test_") and isinstance(fn, types.FunctionType):
                try:
                    fn()
                    ran += 1
                except Exception:
                    fails += 1
                    print(f"FAIL {m}.{name}")
                    traceback.print_exc()
    print(f"\nRAN {ran} tests, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
