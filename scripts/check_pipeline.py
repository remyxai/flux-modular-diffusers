#!/usr/bin/env python3
"""Static checks for a pipelines/<name>/ dir (CI-safe: no FLUX/GPU download).
Usage: python scripts/check_pipeline.py pipelines/<name>"""
import json, sys, py_compile, pathlib
d = pathlib.Path(sys.argv[1]); errs = []
req = ["block.py", "modular_config.json", "modular_model_index.json", "README.md", "e2e.ipynb"]
for f in req:
    if not (d / f).exists(): errs.append(f"missing {f}")
if (d / "block.py").exists():
    try: py_compile.compile(str(d / "block.py"), doraise=True)
    except py_compile.PyCompileError as e: errs.append(f"block.py compile: {str(e)[:120]}")
if (d / "modular_config.json").exists():
    cfg = json.load(open(d / "modular_config.json"))
    am = cfg.get("auto_map", {}).get("ModularPipelineBlocks", "")
    if not am.startswith("block."): errs.append("modular_config auto_map must be block.<Class>")
    cls = am.split(".")[-1]
    if (d / "block.py").exists() and f"class {cls}(" not in (d / "block.py").read_text():
        errs.append(f"class {cls} not found in block.py")
    if (d / "modular_model_index.json").exists():
        idx = json.load(open(d / "modular_model_index.json"))
        if idx.get("_blocks_class_name") != cls: errs.append("_blocks_class_name != config class")
rd = (d / "README.md").read_text().lower() if (d / "README.md").exists() else ""
for sec in ["usage", "how it works", "attribution"]:
    if sec not in rd: errs.append(f"README missing '{sec}' section")
if "citation" not in rd and "references" not in rd:
    errs.append("README missing a 'citation' or 'references' section")
if errs:
    print(f"FAIL {d}:"); [print("  -", e) for e in errs]; sys.exit(1)
print(f"OK {d}")
