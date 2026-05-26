"""FinAcumen acceptance tests (artifacts only under ``test/.artifacts/``).

**Tier A — default CI / no API keys**: import, ``paths``, ``finacumen-dser-race --help``,
optional dry-run when ``datasets/`` is present under the repo root.
No calls to DashScope / LLM in Tier A unless dry-run unexpectedly touches the network.

**Tier B — developer machine**: set ``FINACUMEN_SMOKE_API=1`` and provide valid
API credentials in ``.env`` or ``finacumen/.env`` (never commit secrets).
Requires valid ``configs/config.toml`` with embedding/LLM settings.

Run from repository root::

    pip install -e ./finacumen    # optional if you use test/conftest.py path bootstrap
    pip install pytest
    pytest -c test/pytest.ini test/test_finacumen_acceptance.py -v              # Tier A

    FINACUMEN_SMOKE_API=1 pytest -c test/pytest.ini test/ -v -m api               # Tier B only

Outputs (gitignored): ``test/.artifacts/``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FINACUMEN_ROOT = REPO_ROOT / "finacumen"
_DOTENV_HINT = REPO_ROOT / ".env"
_ARTIFACT_ROOT = Path(__file__).resolve().parent / ".artifacts"


@pytest.fixture(autouse=True)
def _ensure_artifacts_dir() -> None:
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _tier_b_dotenv() -> None:
    """Load sibling .env files before config-backed imports (Tier B hooks only where used)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (REPO_ROOT / ".env", FINACUMEN_ROOT / ".env"):
        load_dotenv(p, override=False)


def test_import_finacumen_ft_config() -> None:
    import finacumen.ft.config  # noqa: F401


def test_paths_memory_constants() -> None:
    from finacumen.ft.paths import (
        DEFAULT_MEMORY_BANK_DIR,
        FINACUMEN_PROJECT_ROOT,
        MEMORY_ROOT,
        REPO_ROOT,
    )

    assert REPO_ROOT.is_absolute()
    assert FINACUMEN_PROJECT_ROOT.is_absolute()
    assert MEMORY_ROOT.is_absolute()
    assert "memory" in str(MEMORY_ROOT).replace("\\", "/")
    assert DEFAULT_MEMORY_BANK_DIR.name == "main"
    assert DEFAULT_MEMORY_BANK_DIR.parent == MEMORY_ROOT


def _finacumen_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    package_root = str(FINACUMEN_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
    return env


def test_config_loads_from_finacumen_without_printing_secrets() -> None:
    from finacumen.ft.config import config
    from finacumen.ft.paths import FINACUMEN_PROJECT_ROOT

    assert config.root_path == FINACUMEN_PROJECT_ROOT
    assert config.workspace_root == FINACUMEN_PROJECT_ROOT / "workspace"
    assert "default" in config.llm
    assert config.llm["default"].model
    assert config.llm["default"].base_url
    if config.embedding_config is not None:
        assert config.embedding_config.provider in {"dashscope", "nv_embed_v2"}


def test_benchmark_dser_race_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "finacumen.ft.benchmark_dser_race", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=_finacumen_subprocess_env(),
    )
    assert r.returncode == 0, r.stderr
    combined = (r.stdout or "") + (r.stderr or "")
    assert "FinAcumen benchmark harness" in combined
    (_ARTIFACT_ROOT / "benchmark_dser_race_help.txt").write_text(
        combined, encoding="utf-8"
    )


def test_benchmark_dser_race_dry_run_when_datasets_present() -> None:
    test_json = REPO_ROOT / "datasets" / "test"
    train_json = REPO_ROOT / "datasets" / "BizBench"
    if not test_json.exists() and not train_json.exists():
        pytest.skip("datasets/ not populated — dry-run Tier A deferred")
    out_dir = _ARTIFACT_ROOT / "dry_run_finacumen_limit_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "finacumen.ft.benchmark_dser_race",
            "--variant",
            "finacumen",
            "--dataset",
            "bizbench",
            "--limit",
            "1",
            "--dry-run",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=_finacumen_subprocess_env(),
    )
    trace = (_ARTIFACT_ROOT / "benchmark_dser_race_dry_run_stdout.txt").write_text(
        (r.stdout or "")
        + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""),
        encoding="utf-8",
    )
    assert trace is not None
    assert r.returncode == 0, (
        _ARTIFACT_ROOT / "benchmark_dser_race_dry_run_stdout.txt"
    )


@pytest.mark.api
def test_embed_text_smoke_tier_b() -> None:
    if os.environ.get("FINACUMEN_SMOKE_API", "").strip() != "1":
        pytest.skip(
            "Tier B: export FINACUMEN_SMOKE_API=1 and provide valid "
            "API credentials in .env"
        )
    async def _run() -> None:
        from finacumen.embeddings import embed_text, embedding_dimension

        dim = embedding_dimension()
        v = await embed_text("tier-b-finacumen-acceptance-vector")
        assert v.shape == (dim,), v.shape

    asyncio.run(_run())
