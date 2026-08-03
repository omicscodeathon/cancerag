"""APA author-date conversion for the superscript-cited Nature manuscript.

Superscript citation run = one or more superscript digits, with normal commas
between groups ("¹⁸,¹⁹") and superscript-minus for ranges ("²⁻⁴"). Unit
superscripts ("mol⁻¹", "10⁻⁴") start with "⁻" or follow normal digits and are
never matched (a citation run must START with a superscript digit). The author
byline (before the first "## " heading) keeps its superscript affiliations.
"""
from __future__ import annotations
import json, re, sys

MD = "manuscript/CancerAg_NatComms.md"
O = json.load(open("scripts/apa_out.json"))
LABEL = {int(n): O[n]["label"] for n in O}
YEAR  = {int(n): O[n]["year"]  for n in O}
SORT  = {int(n): O[n]["sort"]  for n in O}

SUP = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}
SD = "".join(SUP)            # superscript digits
MIN = "⁻"

txt = open(MD).read()
lines = txt.split("\n")
first_h2 = next(i for i,l in enumerate(lines) if l.startswith("## "))
byline = "\n".join(lines[:first_h2])
body   = "\n".join(lines[first_h2:])

# De-italicise "et al." (APA: not italic) so narrative handling sees plain text.
body = body.replace("*et al.*", "et al.")

def to_nums(run):
    """'¹⁸,¹⁹' -> [18,19]; '²⁻⁴' -> [2,3,4]."""
    out=[]
    for grp in run.split(","):
        grp=grp.strip()
        if MIN in grp:
            a,b=grp.split(MIN)
            a="".join(SUP[c] for c in a); b="".join(SUP[c] for c in b)
            out+=list(range(int(a),int(b)+1))
        elif grp and all(c in SUP for c in grp):
            out.append(int("".join(SUP[c] for c in grp)))
    return out

def paren_group(nums):
    nums=sorted(set(nums), key=lambda n:(SORT.get(n,""),YEAR.get(n,"")))
    return "(" + "; ".join(LABEL[n] for n in nums) + ")"

# Superscript-run regex: starts with a superscript digit; groups joined by a
# normal comma or a superscript minus that is itself followed by more sup digits.
RUN = rf"[{SD}]+(?:(?:,|{MIN})[{SD}]+)*"

# 1. ")<run>" right after a "...YYYY)" author-date already in prose -> drop run.
body = re.sub(rf"(\([A-Z][^()]*?,\s*\d{{4}}\))(?:{RUN})", r"\1", body)

# 2. Narrative "et al.<run>" -> "et al. (YYYY)" (single-ref runs only).
def narr(m):
    nums=to_nums(m.group(1))
    if len(nums)==1 and nums[0] in YEAR:
        return f"et al. ({YEAR[nums[0]]})"
    return m.group(0)
body = re.sub(rf"et al\.({RUN})", narr, body)

# 3. All remaining runs -> parenthetical group, with a leading space.
#    The (?<!⁻) lookbehind protects unit superscripts: a citation run never
#    starts immediately after a superscript-minus ("mol⁻¹", "10⁻⁴"), whereas a
#    citation range starts with its first digit ("²⁻⁴"), not the minus.
def grp(m):
    nums=to_nums(m.group(0))
    if not nums or any(n not in LABEL for n in nums):
        return m.group(0)
    return " " + paren_group(nums)
body = re.sub(rf"(?<!{MIN}){RUN}", grp, body)
body = re.sub(r"  +\(", " (", body)

# 4. Alphabetical APA reference list.
new = byline + "\n" + body
head,_,_ = new.partition("\n## References")
apa_sorted = sorted(O.keys(), key=lambda n:(O[n]["sort"],O[n]["year"]))
ref=["","## References",""]
for n in apa_sorted: ref.append(O[n]["apa"]); ref.append("")
new = head + "\n" + "\n".join(ref)
open(MD,"w").write(new)

left = re.findall(RUN, body)
print(f"{MD}: done. remaining superscript runs in body: {len(left)} {left[:10]}")
