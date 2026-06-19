#!/usr/bin/env python3
"""Post-install fixups applied to the active environment.

Run this from any build (build_uv.py, the Dockerfiles, ...) *after* the
relevant packages are installed.  Every fixup is idempotent and degrades to a
warning if the targeted package/line is absent, so it's safe to run more than
once and across environments that don't have every package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def relax_transformers_tokenizers_pin():
    """Allow the source-built tokenizers (0.23.1) under transformers.
    """
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.submodule_search_locations:
        print("transformers not installed; skipping tokenizers pin patch")
        return
    pkg_dir = Path(list(spec.submodule_search_locations)[0])
    table = pkg_dir / "dependency_versions_table.py"
    if not table.is_file():
        print(f"{table} not found; skipping tokenizers pin patch")
        return
    old = '"tokenizers": "tokenizers>=0.22.0,<=0.23.0"'
    new = '"tokenizers": "tokenizers>=0.22.0,<0.24"'
    text = table.read_text()
    if old in text:
        table.write_text(text.replace(old, new))
        print(f"Patched transformers tokenizers pin in {table}")
    elif new in text:
        print("transformers tokenizers pin already patched")
    else:
        print(
            "warning: transformers tokenizers pin line not found "
            f"in {table}; transformers may have changed it"
        )


def main():
    relax_transformers_tokenizers_pin()


if __name__ == "__main__":
    main()
