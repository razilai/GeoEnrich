#!/usr/bin/env python3
"""Render the paper's tables, methods text and figures from `report`'s outputs (see src/paper.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paper  # noqa: E402

if __name__ == "__main__":
    paper.main()
