"""What the interaction fingerprints actually record: a census of every contact."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt

from . import _style as S
from ._data import Artefacts

NAME = "fig16_contact_census"
TITLE = "The interaction fingerprint records occupancy, not pathway preference"
DESCRIPTION = """
Every contact ProLIF records across the 587 docked pairs, counted from the
regenerated fingerprint matrix. (a) Contacts by interaction type, all nine
configured types shown on a log axis. Van der Waals and hydrophobic contacts
carry 92.7 % of the total; the directional, chemically specific interactions a
pharmacological mechanism would be written in are rare, down to a single
cation–π contact in the whole dataset. Neither ionic type fires once, which is
a property of the representation rather than of the biology: only 15 of the
538 curated ligands carry any formal charge, so the neutral structures that
went into docking cannot form the salt bridge that an aminergic ligand makes
at position 3.32. (b) The fifteen Ballesteros–Weinstein positions carrying
most contacts; the twenty busiest of the 148 occupied positions carry 8,838 of
11,545 BW contacts (76.6 %), and they are the canonical orthosteric pocket.
(c) Where the contacts sit: 376 BW columns over 148 generic positions carry
11,545 contacts, while the 243 accession-qualified columns — contacts with no
generic position, so comparable across no two receptors — carry 687 between
them, 2.8 per column. A block of this composition is a record of where a
ligand sits in a pocket, not of which transducer the complex favours, which is
the representation-level reason the structural arm adds nothing measurable.
"""
SOURCES = [
    "processed/regen_20261123/contact_census.json",
    "processed/regen_20261123/interaction_fingerprints.parquet",
    "processed/regen_20261123/unified_ligands.csv",
]

TYPE_LABEL = {
    "VdWContact": "van der Waals",
    "Hydrophobic": "Hydrophobic",
    "HBAcceptor": "H-bond acceptor",
    "HBDonor": "H-bond donor",
    "PiStacking": "π-stacking",
    "PiCation": "π–cation",
    "CationPi": "cation–π",
    "Anionic": "Anionic (ionic)",
    "Cationic": "Cationic (ionic)",
}
CONFIGURED = ["VdWContact", "Hydrophobic", "HBAcceptor", "PiStacking", "HBDonor",
              "PiCation", "CationPi", "Anionic", "Cationic"]
NONSPECIFIC = {"VdWContact", "Hydrophobic"}


def _panel_a(ax, census, facts):
    counts = census["contacts_by_interaction_type"]
    total = census["total_contacts"]
    ys = range(len(CONFIGURED))
    for y, t in zip(ys, CONFIGURED):
        n = int(counts.get(t, 0))
        col = S.GREY_DARK if t in NONSPECIFIC else S.BLUE
        ax.barh(len(CONFIGURED) - 1 - y, max(n, 0.6), color=col, height=0.66, zorder=2)
        lab = f"{n:,}" if n else "never fires"
        ax.text(max(n, 0.6) * 1.35, len(CONFIGURED) - 1 - y, lab, va="center",
                ha="left", fontsize=6.3,
                color=S.TEXT_PRIMARY if n else S.TEXT_MUTED, zorder=4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([TYPE_LABEL[t] for t in reversed(CONFIGURED)])
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 40000)
    ax.set_ylim(-1.9, len(CONFIGURED) - 0.4)
    ax.set_xlabel("Contacts (log scale)")
    S.xgrid(ax)
    share = census["nonspecific_share_vdw_plus_hydrophobic"]
    ns = sum(int(counts.get(t, 0)) for t in NONSPECIFIC)
    ax.text(0.7, -1.5,
            f"van der Waals + hydrophobic:\n{ns:,} of {total:,} contacts ({share:.1%})",
            ha="left", va="center", fontsize=6.4, color=S.TEXT_PRIMARY, zorder=5)
    facts["total contacts"] = total
    facts["contacts by type"] = {TYPE_LABEL[t]: int(counts.get(t, 0))
                                 for t in CONFIGURED}
    facts["non-specific share (vdW + hydrophobic)"] = share
    facts["interaction types configured / observed"] = (
        f"{census['interactions_configured']} / {census['interaction_types_observed']}"
    )


def _panel_b(ax, art, census, facts, n_show=15):
    df = art.parquet(SOURCES[1])
    bw_cols = [c for c in df.columns if c.startswith("ifp_BW")]
    sums = df[bw_cols].sum()
    pos: Counter = Counter()
    for col, v in sums.items():
        m = re.match(r"ifp_BW(.+?)_(?:[A-Za-z]+)$", col)
        pos[m.group(1) if m else col] += int(v)
    top = pos.most_common(n_show)
    labels = [p for p, _ in top][::-1]
    values = [v for _, v in top][::-1]
    cmax = max(values)
    for i, (lab, v) in enumerate(zip(labels, values)):
        ax.barh(i, v, color=S.cmap_hex(S.SEQ_CMAP, 0.35 + 0.5 * v / cmax),
                height=0.68, zorder=2)
        ax.text(v + cmax * 0.02, i, f"{v:,}", va="center", ha="left", fontsize=6.1,
                color=S.TEXT_SECONDARY, zorder=4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, cmax * 1.28)
    ax.set_ylim(-2.2, n_show - 0.4)
    ax.set_xlabel("Contacts at this generic position")
    ax.set_ylabel("Ballesteros–Weinstein position")
    S.xgrid(ax)
    top20 = sum(v for _, v in pos.most_common(20))
    bw_total = census["ballesteros_weinstein"]["contacts"]
    ax.text(cmax * 0.02, -1.6,
            f"20 of {len(pos)} occupied positions carry\n{top20:,} of {bw_total:,} "
            f"BW contacts ({top20 / bw_total:.1%})",
            ha="left", va="center", fontsize=6.4, color=S.TEXT_PRIMARY, zorder=5)
    facts["distinct BW positions occupied"] = len(pos)
    facts["top 20 BW positions"] = (
        f"{top20:,} of {bw_total:,} BW contacts ({top20 / bw_total:.1%})"
    )
    facts["busiest positions"] = ", ".join(f"{p} ({v:,})" for p, v in top[:6])


def _panel_c(ax, census, facts):
    bw = census["ballesteros_weinstein"]
    q = census["receptor_qualified"]
    groups = [
        ("BW\nposition", bw["n_columns"], bw["contacts"], S.BLUE),
        ("Accession\nqualified", q["n_columns"], q["contacts"], S.ORANGE),
    ]
    rates = [ncon / ncol for _, ncol, ncon, _ in groups]
    for i, ((lab, ncol, ncon, col), rate) in enumerate(zip(groups, rates)):
        ax.bar(i, rate, width=0.52, color=col, zorder=2)
        ax.text(i, rate + max(rates) * 0.035, f"{rate:.1f}", ha="center", va="bottom",
                fontsize=7.4, fontweight="bold", color=S.TEXT_PRIMARY, zorder=4)
        ax.text(i, rate + max(rates) * 0.16, f"{ncol} columns\n{ncon:,} contacts",
                ha="center", va="bottom", fontsize=5.9, color=S.TEXT_SECONDARY,
                zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.65, len(groups) - 0.35)
    ax.set_ylim(0, max(rates) * 1.45)
    ax.set_ylabel("Contacts per column")
    S.ygrid(ax)
    facts["BW columns / positions / contacts"] = (
        f"{bw['n_columns']} / {bw['n_distinct_positions']} / {bw['contacts']:,}"
    )
    facts["accession-qualified columns / contacts"] = (
        f"{q['n_columns']} / {q['contacts']:,} ({rates[1]:.1f} per column)"
    )


def _charged_ligands(art: Artefacts) -> str:
    """Why neither ionic interaction type can fire: the ligands are neutral."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    smiles = art.csv(SOURCES[2]).canonical_smiles_std.dropna().unique()
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None]
    charged = sum(any(a.GetFormalCharge() for a in m.GetAtoms()) for m in mols)
    return f"{charged} of {len(mols)} curated ligands"


def build(art: Artefacts):
    census = json.loads(art.path(SOURCES[0]).read_text())
    facts: dict = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 0.58],
                                          "wspace": 0.50})
    _panel_a(axes[0], census, facts)
    _panel_b(axes[1], art, census, facts)
    _panel_c(axes[2], census, facts)
    for ax, letter, x in zip(axes, "abc", (-0.42, -0.22, -0.46)):
        S.panel_label(ax, letter, x=x)
    facts["fingerprint sha256 (first 8)"] = census["source_sha256"][:8]
    facts["pairs / ifp columns"] = f"{census['n_rows']} / {census['n_ifp_columns']}"
    facts["mean contacts per pair"] = census["contacts_per_pair_mean"]
    facts["ligands carrying a formal charge"] = _charged_ligands(art)
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
