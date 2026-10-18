"""Build every manuscript figure and write ``FIGURES.md`` beside them.

    PYTHONPATH=src uv run python -m scripts.figures [--out DIR] [--data DIR]
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import RETIRED, all_modules
from ._cli import build_module, parser
from ._data import REPO_ROOT, Artefacts, MissingArtefact

MANUSCRIPT_NAME = "CancerAg_NatComms_IMRaD.md"


def manuscript_figure_numbers(out: Path) -> dict[str, str]:
    """Map figure basename -> 'Figure N' from the manuscript's <!--fig:...--> markers.

    The manuscript is local-only (not committed); look beside the output
    directory first, then in this checkout.
    """
    candidates = [Path(out).resolve().parent / MANUSCRIPT_NAME,
                  REPO_ROOT / "manuscript" / MANUSCRIPT_NAME]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}
    text = path.read_text()
    out: dict[str, str] = {}
    for m in re.finditer(r"\*\*Figure (\d+) \|.*?\n<!--fig:([^>]+)-->", text):
        for f in m.group(2).split(","):
            out[Path(f.strip()).stem] = f"Figure {m.group(1)}"
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    return str(v)


def write_index(records: list[dict], skipped: list[tuple[str, str]], out: Path,
                data_dir: Path) -> Path:
    numbers = manuscript_figure_numbers(out)
    lines = [
        "# Manuscript figures",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by "
        "`PYTHONPATH=src uv run python -m scripts.figures` "
        f"from artefacts under `{data_dir}`.",
        "Each figure is written as a 300 dpi PNG and a vector PDF. Source paths "
        "are relative to `data/`. Key numbers are read from the artefacts at "
        "build time, not typed in.",
        "",
        "| File | Manuscript | What it shows |",
        "|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| `{r['name']}` | {numbers.get(r['name'], '—')} | {r['title']} |"
        )
    lines.append("")
    for r in records:
        lines += [
            f"## `{r['name']}` — {r['title']}",
            "",
            f"Manuscript: {numbers.get(r['name'], 'not referenced')}",
            "",
            r["description"],
            "",
            "**Source artefacts**",
            "",
            *[f"- `{s}` (modified {r['source_dates'][s]})" for s in r["sources"]],
            "",
            "**Key numbers shown**",
            "",
            *[f"- {k}: {_fmt(v)}" for k, v in r["facts"].items()],
            "",
        ]
    if skipped:
        lines += ["## Not built", ""]
        lines += [f"- `{n}`: {why}" for n, why in skipped]
        lines.append("")
    if RETIRED:
        lines += ["## Retired from the 2026-08-25 set", ""]
        lines += [f"- `{n}`: {why}" for n, why in RETIRED.items()]
        lines.append("")
    p = out / "FIGURES.md"
    p.write_text("\n".join(lines))
    (out / "figures_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str)
    )
    return p


def main() -> int:
    args = parser("Build every CancerAg manuscript figure").parse_args()
    art = Artefacts(args.data) if args.data else Artefacts()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    records, skipped = [], []
    for mod in all_modules():
        try:
            records.append(build_module(mod, art, out))
            print(f"[ok]   {mod.NAME}")
        except MissingArtefact as exc:
            skipped.append((mod.NAME, str(exc)))
            print(f"[skip] {mod.NAME}: {exc}", file=sys.stderr)

    for name, why in RETIRED.items():
        for ext in ("png", "pdf"):
            stale = out / f"{name}.{ext}"
            if stale.exists():
                stale.unlink()
                print(f"[rm]   {stale.name} (retired: {why})")

    idx = write_index(records, skipped, out, art.data_dir)
    print(f"[ok]   {idx}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
