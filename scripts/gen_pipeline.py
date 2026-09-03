#!/usr/bin/env python3
"""Generate a Modular-Diffusers pipeline dir from a recipe.
    python scripts/gen_pipeline.py <recipe-name> <out-dir>
"""
import sys
from flux_modular.recipes import load_recipes
from flux_modular.codegen import generate_pipeline

def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    name, out = sys.argv[1], sys.argv[2]
    recipes = load_recipes()
    if name not in recipes:
        print(f"unknown recipe {name!r}; have: {sorted(recipes)}"); sys.exit(1)
    print("generated", generate_pipeline(recipes[name], out))

if __name__ == "__main__":
    main()
