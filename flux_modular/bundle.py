"""Generate the flat, vendorable ``flux_modular.py`` (plumbing + attention concatenated).

``trust_remote_code`` loads only FLAT sibling ``.py`` files, so each pipeline HF repo ships this bundled
file beside its ``block.py`` (which does ``from .flux_modular import ...``). Both source files are
self-contained (no cross-import), so concatenation is safe.

    python -m flux_modular.bundle > pipelines/<name>/flux_modular.py
"""

import os

_HERE = os.path.dirname(__file__)
_HEADER = (
    '"""flux_modular (bundled flat build) — GENERATED from flux_modular/ by bundle.py; do not edit here.\n'
    'Vendored beside block.py so trust_remote_code loads it as a flat sibling."""\n'
)


def build():
    parts = [_HEADER]
    for f in ("plumbing.py", "attention.py"):
        parts.append(f"\n# ============================== {f} ==============================\n")
        parts.append(open(os.path.join(_HERE, f)).read())
    return "".join(parts)


if __name__ == "__main__":
    print(build())
