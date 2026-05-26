"""Repository layout anchors for the installable FinAcumen package."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINACUMEN_PROJECT_ROOT = REPO_ROOT / "finacumen"

# V2 memory root — contains one subdirectory per bank version
# (e.g., memory/baseline/, memory/strategy_a/)
MEMORY_ROOT = FINACUMEN_PROJECT_ROOT / "memory"

# Default bank for --memory-dir
DEFAULT_MEMORY_BANK_DIR = MEMORY_ROOT / "main"
