"""Event-triggered action chunking for ManiSkill diffusion policies."""

from __future__ import annotations

import os
import platform
from pathlib import Path

# In WSL2, Mesa software Vulkan ICD is needed for CPU-based rendering and material initialization.
if "microsoft" in platform.release().lower():
    mesa_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    if mesa_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(mesa_icd))
