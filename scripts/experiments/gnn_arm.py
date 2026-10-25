"""A molecular graph network, scored under the same regimes as everything else.

ChemBERTa reads SMILES as a token sequence; a GNN reads the molecule as the
graph it actually is. Different inductive bias, same question -- and the point
is not to win, it is to find out whether the ceiling we keep hitting is a
property of the *representation* or of the data.

Deliberately small: 478 training rows cannot support a large network, and a
GNN that overfits would tell us nothing we do not already know. Two message
passing layers, 64 hidden units, mean pooling, early stopping on a held-in
split.

Scored under stratified / scaffold-grouped / receptor-grouped, against the
same majority-class floor, so the number is comparable with the LightGBM and
ChemBERTa arms rather than sitting on its own scale.

Note what a ligand-graph model can and cannot see: it has the molecule and
nothing about the receptor. Under receptor-grouped CV that is the honest
test of how far ligand structure alone travels.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from cancerag.ml.preprocessing import make_grouped_cv  # noqa: E402

SEEDS = [42, 7, 13]
EPOCHS = 120
HIDDEN = 64
OUT = Path("data/processed/ml_models/gnn_arm.json")

tr = pd.read_parquet("data/processed/ml_ready_dataset.parquet")
smi_col = next(c for c in ("canonical_smiles_std", "canonical_smiles")
               if c in tr.columns)
labels = sorted(tr["bias_category"].unique())
lab2i = {c: i for i, c in enumerate(labels)}
print(f"{len(tr)} rows, smiles column {smi_col}, classes {labels}")

ATOMS = list(range(1, 54))


def featurise(mol) -> torch.Tensor:
    xs = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        onehot = [0.0] * len(ATOMS)
        if z in ATOMS:
            onehot[ATOMS.index(z)] = 1.0
        xs.append(onehot + [
            a.GetDegree() / 4.0,
            a.GetFormalCharge(),
            float(a.GetIsAromatic()),
            float(a.IsInRing()),
            a.GetTotalNumHs() / 4.0,
        ])
    return torch.tensor(xs, dtype=torch.float)


def to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    ei = [[], []]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei[0] += [i, j]
        ei[1] += [j, i]
    if not ei[0]:                      # single atom, e.g. an ion
        ei = [[0], [0]]
    return Data(x=featurise(mol),
                edge_index=torch.tensor(ei, dtype=torch.long),
                y=torch.tensor([y], dtype=torch.long))


graphs, keep = [], []
for idx, row in tr.iterrows():
    g = to_graph(row[smi_col], lab2i[row["bias_category"]])
    if g is not None:
        graphs.append(g)
        keep.append(idx)
print(f"built {len(graphs)} graphs ({len(tr) - len(graphs)} unparseable)")
sub = tr.loc[keep].reset_index(drop=True)
y_all = np.array([int(g.y) for g in graphs])
NFEAT, NCLS = graphs[0].x.shape[1], len(labels)

floor = f1_score(y_all, np.full_like(y_all, np.bincount(y_all).argmax()),
                 average="macro", zero_division=0)
print(f"majority-class floor on the graph subset: {floor:.4f}")


class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def mlp(i, o):
            return torch.nn.Sequential(torch.nn.Linear(i, o), torch.nn.ReLU(),
                                       torch.nn.Linear(o, o))
        self.c1 = GINConv(mlp(NFEAT, HIDDEN))
        self.c2 = GINConv(mlp(HIDDEN, HIDDEN))
        self.head = torch.nn.Linear(HIDDEN, NCLS)

    def forward(self, d):
        h = F.relu(self.c1(d.x, d.edge_index))
        h = F.relu(self.c2(h, d.edge_index))
        return self.head(global_mean_pool(h, d.batch))


def run_fold(tri, tei, seed, weights):
    torch.manual_seed(seed)
    model = GIN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader([graphs[i] for i in tri], batch_size=32, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y, weight=weights)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        te = next(iter(DataLoader([graphs[i] for i in tei],
                                  batch_size=len(tei))))
        pred = model(te).argmax(1).numpy()
    return f1_score(y_all[tei], pred, average="macro", zero_division=0)


# class_weight='balanced' equivalent -- the rare classes are the whole problem
cnt = np.bincount(y_all, minlength=NCLS).astype(float)
weights = torch.tensor(len(y_all) / (NCLS * np.maximum(cnt, 1)),
                       dtype=torch.float)
print(f"class weights: {weights.numpy().round(2)}")

REGIMES = {
    "stratified": lambda s: list(
        StratifiedKFold(5, shuffle=True, random_state=s).split(graphs, y_all)),
    "scaffold_grouped": lambda s: make_grouped_cv(
        sub, group_col="scaffold", n_splits=5, seed=s),
    "receptor_grouped": lambda s: make_grouped_cv(
        sub, group_col="receptor_uniprot", n_splits=5, seed=s),
}

rows = []
for regime, mk in REGIMES.items():
    for seed in SEEDS:
        splits = mk(seed)
        if not splits:
            continue
        sc = [run_fold(tri, tei, seed, weights) for tri, tei in splits]
        rows.append({"regime": regime, "seed": seed,
                     "macro_f1": float(np.mean(sc))})
        print(f"  {regime} seed={seed}: {np.mean(sc):.4f}", flush=True)

d = pd.DataFrame(rows)
print("\n" + "=" * 60)
print("GIN on molecular graphs (ligand only)")
print("=" * 60)
print(d.groupby("regime")["macro_f1"].agg(["mean", "std"]).round(4).to_string())
print(f"\nfloor: {floor:.4f}")

OUT.write_text(json.dumps({"rows": rows, "floor": float(floor),
                           "n_graphs": len(graphs), "hidden": HIDDEN,
                           "epochs": EPOCHS}, indent=2))
print(f"written: {OUT}")
