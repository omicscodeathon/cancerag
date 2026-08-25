"""Build a self-contained HTML reading preview with figures placed inline.

The DOCX builder embeds figures only under the figure-legend sections at the
end, which is the Nature submission convention but makes the manuscript hard to
read: a Results paragraph cites "Fig. 3" and the reader has to scroll past the
Methods to see it. The markdown carries no image tags at all, so a plain
markdown preview shows no figures whatever.

This renderer places each figure at its first in-text mention in Results, keeps
the legend text with it, and inlines every image as a data URI so the file works
with no network and no sibling asset directory.

Run:
    PYTHONPATH=src python scripts/build_preview_html.py <input.md> <output.html>
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import mistune

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "manuscript" / "figures"

def figure_files(md: str) -> dict[int, list[str]]:
    """Figure number -> image files, read from each legend's <!--fig:...--> tag.

    The mapping travels with the legend so a renumbered document stays correct
    without a second table to keep in step.
    """
    out: dict[int, list[str]] = {}
    for m in re.finditer(r"\*\*Figure (\d+) \|(?:(?!\*\*Figure )[\s\S])*?<!--fig:([^>]+?)-->", md):
        out[int(m.group(1))] = [f.strip() for f in m.group(2).split(",") if f.strip()]
    return out


def data_uri(name: str) -> str | None:
    p = FIG_DIR / name
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def legends(md: str) -> dict[int, tuple[str, str]]:
    """Figure number -> (title, remaining legend text)."""
    out: dict[int, tuple[str, str]] = {}
    for m in re.finditer(r"^\*\*Figure (\d+) \|(.+?)\*\*(.*?)$", md, re.M | re.S):
        out[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return out


def figure_block(num: int, leg: dict, files: dict) -> str:
    imgs = [u for u in (data_uri(f) for f in files.get(num, [])) if u]
    if not imgs:
        return ""
    title, rest = leg.get(num, ("", ""))
    cap = f"<b>Figure {num}</b>"
    if title:
        cap += f" | {mistune.html(title).strip()[3:-4]}"
    rest = re.sub(r"<!--fig:[^>]*?-->", "", rest).strip()
    body = f"<div class='legend-body'>{mistune.html(rest)}</div>" if rest else ""
    tags = "".join(f"<img src='{u}' alt='Figure {num}'>" for u in imgs)
    return (f"<figure class='inline-fig' id='fig-{num}'>{tags}"
            f"<figcaption>{cap}{body}</figcaption></figure>")


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    md = src.read_text()
    leg = legends(md)
    files = figure_files(md)

    # Render body and legend sections separately: the legend sections keep their
    # figures too, so the end-of-paper layout a reviewer expects still works.
    html = mistune.html(re.sub(r"<!--fig:[^>]*?-->", "", md))

    # Place each figure at its first in-text citation. Anchor on the closing tag
    # of the paragraph containing the mention so a figure never lands mid-sentence.
    placed: set[int] = set()

    def place(match: re.Match) -> str:
        para = match.group(0)
        add = []
        for mm in re.finditer(r"Fig(?:ure)?\.?\s*(\d+)", para):
            num = int(mm.group(1))
            if num in placed:
                continue
            blk = figure_block(num, leg, files)
            if blk:
                placed.add(num)
                add.append(blk)
        return para + "".join(add)

    # Paragraphs and headings, and only before the legend sections. Headings
    # must be scanned too: several Results sections cite their figure only in
    # the heading ("... (Fig. 5)"), so a paragraph-only pass drops the two
    # main figures the paper is built around.
    cut = html.find("<h2>Figure legends")
    head, tail = (html[:cut], html[cut:]) if cut != -1 else (html, "")
    head = re.sub(r"<(p|h2|h3)>(?:(?!</\1>).)*?</\1>", place, head, flags=re.S)

    # legend sections: show the image under each legend as well
    def legend_img(match: re.Match) -> str:
        para = match.group(0)
        mm = re.search(r"<strong>Figure (\d+) \|", para) or re.search(r"<b>Figure (\d+)", para)
        if not mm:
            return para
        num = int(mm.group(1))
        tags = "".join(f"<img src='{u}'>" for u in
                       (data_uri(f) for f in files.get(num, [])) if u)
        return para + (f"<figure class='legend-fig'>{tags}</figure>" if tags else "")

    tail = re.sub(r"<p>(?:(?!</p>).)*?</p>", legend_img, tail, flags=re.S)

    title = re.search(r"^# (.+)$", md, re.M)
    title = title.group(1) if title else src.stem

    dst.write_text(TEMPLATE.replace("{{TITLE}}", title)
                           .replace("{{BODY}}", head + tail)
                           .replace("{{NFIG}}", str(len(placed))))
    print(f"{dst}  ({dst.stat().st_size/1_000_000:.1f} MB, {len(placed)} figures placed inline)")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root{ --bg:#fbfaf8; --fg:#1a1a18; --muted:#5c5a55; --rule:#dedbd4;
         --accent:#7a2e2e; --card:#fff; }
  @media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
    --bg:#16161a; --fg:#e8e6e1; --muted:#a3a099; --rule:#33323a;
    --accent:#e08b8b; --card:#1e1e24; } }
  :root[data-theme="dark"]{ --bg:#16161a; --fg:#e8e6e1; --muted:#a3a099;
    --rule:#33323a; --accent:#e08b8b; --card:#1e1e24; }
  *{box-sizing:border-box}
  body{background:var(--bg); color:var(--fg); margin:0;
       font:16px/1.66 Charter,"Iowan Old Style",Georgia,serif;
       padding:3rem 1.25rem 6rem;}
  main{max-width:44rem; margin:0 auto;}
  h1{font-size:1.85rem; line-height:1.24; letter-spacing:-.01em; margin:0 0 1.4rem;}
  h2{font-size:1.22rem; margin:3rem 0 .9rem; padding-bottom:.35rem;
     border-bottom:1px solid var(--rule);}
  h3{font-size:1.02rem; margin:2rem 0 .6rem; color:var(--accent);}
  p{margin:0 0 1.05rem;}
  a{color:var(--accent);}
  code{font:.87em ui-monospace,SFMono-Regular,Menlo,monospace;
       background:color-mix(in srgb,var(--fg) 7%,transparent);
       padding:.12em .34em; border-radius:3px;}
  hr{border:0; border-top:1px solid var(--rule); margin:2.5rem 0;}
  blockquote{margin:0 0 1rem; padding-left:1rem; border-left:3px solid var(--rule);
             color:var(--muted);}
  /* wide content scrolls inside its own box, never the page */
  .tablewrap{overflow-x:auto; margin:0 0 1.4rem;}
  table{border-collapse:collapse; font-size:.86rem; min-width:100%;
        font-family:system-ui,-apple-system,Segoe UI,sans-serif;}
  th,td{border:1px solid var(--rule); padding:.4rem .6rem; text-align:left;
        vertical-align:top; white-space:nowrap;}
  th{background:color-mix(in srgb,var(--fg) 6%,transparent); font-weight:600;}
  figure{margin:2rem 0; padding:1rem; background:var(--card);
         border:1px solid var(--rule); border-radius:8px;}
  figure img{width:100%; height:auto; display:block; border-radius:4px;
             background:#fff; margin-bottom:.6rem;}
  figcaption{font-size:.83rem; line-height:1.5; color:var(--muted);
             font-family:system-ui,-apple-system,Segoe UI,sans-serif;}
  figcaption b{color:var(--fg);}
  .legend-body p{margin:.4rem 0 0;}
  .banner{font:0.8rem/1.5 system-ui,sans-serif; color:var(--muted);
          border:1px dashed var(--rule); border-radius:6px; padding:.7rem .9rem;
          margin:0 0 2.5rem;}
</style></head><body><main>
<p class="banner">Reading preview — {{NFIG}} figures placed at their first mention in the text.
The submission DOCX keeps figures under the legend sections at the end, per journal convention.</p>
{{BODY}}
</main>
<script>
  // wrap tables so wide ones scroll on their own instead of the page
  document.querySelectorAll("table").forEach(t => {
    const w = document.createElement("div"); w.className = "tablewrap";
    t.parentNode.insertBefore(w, t); w.appendChild(t);
  });
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
