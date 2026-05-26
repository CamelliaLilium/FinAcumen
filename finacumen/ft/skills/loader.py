"""
Skills loader — emulates Anthropic's three-stage progressive disclosure:
  Stage 1: Metadata scan (YAML frontmatter: name, description, tags)
  Stage 2: Full SKILL.md load (system instruction block)
  Stage 3: Resources (scripts/, templates/) — not used for benchmark QA

Each skill is a directory under `skills/{skill-name}/` containing SKILL.md.
SKILL.md starts with `---` YAML frontmatter, then Markdown body.

The body section titled "## System instruction" (or its immediate `>` blockquote)
is extracted and appended to the DSER system prompt when the skill is routed.

This is a minimal re-implementation of Anthropic's standard (published Dec 2025)
sufficient for our DSER race use-case; upstream-compatible file layout means
skills can be shipped as-is to a real Claude agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml  # pyyaml is a standard dependency for Anthropic skills
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"


@dataclass
class Skill:
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    directory: Optional[Path] = None
    body: str = ""
    system_instruction: str = ""

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, tags={self.tags})"


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from Markdown body."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text = m.group(1)
    body = text[m.end():]
    if _HAS_YAML:
        try:
            data = yaml.safe_load(fm_text) or {}
        except Exception:
            data = _fallback_yaml_parse(fm_text)
    else:
        data = _fallback_yaml_parse(fm_text)
    return data, body


def _fallback_yaml_parse(text: str) -> dict:
    """Minimal YAML subset: key: value, list via [a, b, c]."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            out[key] = [v.strip().strip('"\'') for v in val[1:-1].split(",") if v.strip()]
        else:
            out[key] = val.strip('"\'')
    return out


_INSTRUCTION_SECTION_RE = re.compile(
    r"##\s*System instruction[^\n]*\n(.*?)(?=\n##\s|\Z)", re.DOTALL
)


def _extract_system_instruction(body: str) -> str:
    """Pull text between '## System instruction' and next '## ' (or EOF).
    Strips leading '> ' blockquote markers so it reads as a plain prompt."""
    m = _INSTRUCTION_SECTION_RE.search(body)
    if not m:
        return ""
    raw = m.group(1).strip()
    # If the section body is mostly a blockquote, de-quote it.
    lines_out: list[str] = []
    for line in raw.splitlines():
        ls = line.lstrip()
        if ls.startswith(">"):
            lines_out.append(ls[1:].lstrip())
        else:
            lines_out.append(line)
    text = "\n".join(lines_out).strip()
    # Also cut at the first '```' fenced code-block marker — those are
    # supplementary checklists/schemas, not part of the system instruction.
    fence = text.find("```")
    if fence >= 0:
        text = text[:fence].rstrip()
    return text


def load_skill(path_or_name: str | Path) -> Skill:
    """Load a single skill by directory path or by skill-name."""
    p = Path(path_or_name) if not isinstance(path_or_name, str) else None
    if p is None or not p.exists():
        # Resolve by name
        p = SKILLS_ROOT / str(path_or_name)
    if not p.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {p}")
    skill_md = p / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md missing in: {p}")
    text = skill_md.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    return Skill(
        name=str(fm.get("name", p.name)),
        description=str(fm.get("description", "")),
        tags=fm.get("tags") or [],
        version=str(fm.get("version", "1.0.0")),
        directory=p,
        body=body,
        system_instruction=_extract_system_instruction(body),
    )


def list_skills(root: Optional[Path] = None) -> list[Skill]:
    """Stage-1 metadata scan: return all available skills with lightweight info."""
    r = root or SKILLS_ROOT
    if not r.exists():
        return []
    out = []
    for p in sorted(r.iterdir()):
        if p.is_dir() and (p / "SKILL.md").exists():
            try:
                out.append(load_skill(p))
            except Exception:
                continue
    return out


# ── Routing: map dataset/answer_type → skill name ──────────────────────────

ANSWER_TYPE_TO_SKILL = {
    "mcq":       "financial-mcq-multiselect",
    "numerical": "financial-numerical",
    "free_text": "financial-longform-qa",
    "boolean":   "financial-longform-qa",
}


def route_skill(target: dict) -> Optional[Skill]:
    """Pick a skill for a target based on answer_type.
    Returns None if no suitable skill is registered."""
    atype = target.get("answer_type", "")
    skill_name = ANSWER_TYPE_TO_SKILL.get(atype)
    if not skill_name:
        return None
    try:
        return load_skill(skill_name)
    except FileNotFoundError:
        return None
