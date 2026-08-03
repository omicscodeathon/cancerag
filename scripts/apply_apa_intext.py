"""Convert bracketed numeric citations [N] to APA author-date in-text form, and
replace the numbered reference list with an alphabetical APA list.

Handles, in order:
  - protects the author byline (affiliation markers like **Name**[1, 4])
  - "(Author *et al.*, YYYY)[N]"  -> drop the [N] (author-date already present)
  - "Author *et al.*[N]"          -> "Author et al. (YYYY)"   (narrative)
  - "[N]", "[N, M]", "[N-M]"      -> "(Label; Label; ...)"     (parenthetical,
                                      alphabetical, semicolon-separated)

Usage: python scripts/apply_apa_intext.py <manuscript.md>
"""
from __future__ import annotations
import json, re, sys

MD = sys.argv[1]
O = json.load(open("scripts/apa_out.json"))
LABEL = {int(n): O[n]["label"] for n in O}      # parenthetical inner, e.g. "Hauser et al., 2017"
YEAR  = {int(n): O[n]["year"]  for n in O}
NARR  = {int(n): O[n]["narr"]  for n in O}
SORT  = {int(n): O[n]["sort"]  for n in O}

txt = open(MD).read()
lines = txt.split("\n")

# ---- 1. Protect the byline: lines before the first "## " heading that contain
#         "**Name**[...]" affiliation markers. We blank out citation handling there.
first_h2 = next(i for i,l in enumerate(lines) if l.startswith("## "))
byline_block = "\n".join(lines[:first_h2])
body_block   = "\n".join(lines[first_h2:])

def expand(nums_str):
    """'2-4' or '18, 19' -> sorted unique list of ints."""
    out=[]
    for tok in nums_str.split(","):
        tok=tok.strip()
        if "-" in tok:
            a,b=tok.split("-"); out+=list(range(int(a),int(b)+1))
        elif tok.isdigit():
            out.append(int(tok))
    return out

def paren_group(nums):
    # alphabetical by first-author surname, APA semicolon separation
    nums=sorted(set(nums), key=lambda n: (SORT.get(n,""), YEAR.get(n,"")))
    return "(" + "; ".join(LABEL[n] for n in nums) + ")"

# ---- 1b. A handful of citations were authored with a SPACE before "[" (e.g.
#          "GPCR bias [39, 40]"). Re-attach them so the attachment rule in step 4
#          catches them; genuine non-citations (hyperparameter ranges) are never
#          attached this way.
body_block = body_block.replace("GPCR bias [39, 40]", "GPCR bias[39, 40]")

# ---- 2. "(Author *et al.*, YYYY)[N]" or "(Author, YYYY)[N]" -> drop [N]
body_block = re.sub(r"(\([A-Z][^()]*?,\s*\d{4}\))\[(\d+)\]", r"\1", body_block)

# ---- 3. Narrative "Author *et al.*[N]" / "Surname's[N]" / "Surname[N]" where a
#         capitalised author-ish token immediately precedes the bracket and the
#         bracket is a SINGLE reference -> "Author et al. (YYYY)" style.
def narr_repl(m):
    name, num = m.group(1), int(m.group(2))
    if num not in NARR: return m.group(0)
    # Use the year-in-parens narrative; keep the author text already in prose.
    return f"{name} ({YEAR[num]})"
# matches "Omieczynski *et al.*[10]", "Biopython's[61]", "RDKit's[26]" etc.
body_block = re.sub(r"((?:[A-Z][A-Za-z0-9\-]+(?:'s)?\s+)?\*et al\.\*)\[(\d+)\]", narr_repl, body_block)

# ---- 4. All remaining "[N]" / "[N, M]" / "[N-M]" -> parenthetical group.
#         A real citation is ATTACHED to the preceding token (e.g. "Vina[15, 16]").
#         Hyperparameter ranges / CIs have a SPACE before "[" ("∈ [3, 10]",
#         "0.247 [0.205, 0.292]") and must NOT be touched. The (?<=\S) lookbehind
#         enforces the attachment; every referenced number must also be 1..63.
def grp_repl(m):
    nums=expand(m.group(1))
    if any(n not in LABEL for n in nums):   # any out-of-range int -> not a citation
        return m.group(0)
    if not nums: return m.group(0)
    return " " + paren_group(nums)          # APA: space before the parenthetical
body_block = re.sub(r"(?<=\S)\[(\d+(?:\s*[,-]\s*\d+)*)\]", grp_repl, body_block)
# Collapse any accidental double space introduced before a citation.
body_block = re.sub(r"  +\(", " (", body_block)

# ---- 5. Rebuild the reference list (alphabetical APA) in the References section.
new_text = byline_block + "\n" + body_block
# Replace everything from "## References" onward with the APA alphabetical list.
head, sep, _ = new_text.partition("\n## References")
apa_sorted = sorted(O.keys(), key=lambda n: (O[n]["sort"], O[n]["year"]))
ref_lines = ["", "## References", ""]
for n in apa_sorted:
    ref_lines.append(O[n]["apa"]); ref_lines.append("")
new_text = head + "\n" + "\n".join(ref_lines)

open(MD,"w").write(new_text)

# ---- report
remaining = re.findall(r"(?<!\*)\[\d+(?:\s*[,-]\s*\d+)*\]", body_block)
print(f"{MD}: conversion done.")
print("  remaining bracket-number tokens in body:", len(remaining), remaining[:10])
