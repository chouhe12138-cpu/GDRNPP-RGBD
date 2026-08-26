#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research.diagnostics.pose_structure.runner import main


if __name__ == "__main__":
    main()
