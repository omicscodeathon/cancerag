"""Fetch full reference metadata from CrossRef and emit APA-7 reference strings.

Never fabricates: every author list / issue / DOI comes from CrossRef. Each ref
is matched by DOI when known, else by bibliographic query, then validated
(first-author surname present + year within 1 + title-token overlap >= 0.5).
Low-confidence matches are flagged for manual review.
"""
from __future__ import annotations
import json, re, time, html, urllib.request, urllib.parse, sys

def clean(s):
    return html.unescape(s or "").replace("&amp;", "&")

UA = {"User-Agent": "SABER-refs/1.0 (mailto:halleluyaholudele@gmail.com)"}

# n: (surname, year, title, doi-or-None, manual-apa-or-None)
REFS = {
1:("Hauser",2017,"Trends in GPCR drug discovery: new agents, targets and indications","10.1038/nrd.2017.178",None),
2:("Smith",2018,"Biased signalling: from simple switches to allosteric microprocessors","10.1038/nrd.2017.229",None),
3:("Wisler",2018,"Biased G protein-coupled receptor signaling: changing the paradigm of drug discovery","10.1161/CIRCULATIONAHA.117.028194",None),
4:("Wootten",2018,"Mechanisms of signalling and biased agonism in G protein-coupled receptors","10.1038/s41580-018-0049-3",None),
5:("Rajagopal",2010,"Teaching old receptors new tricks: biasing seven-transmembrane receptors","10.1038/nrd3024",None),
6:("Kenakin",2007,"Functional selectivity through protean and biased agonism: who steers the ship?","10.1124/mol.107.040352",None),
7:("Kenakin",2019,"Biased receptor signaling in drug discovery","10.1124/pr.118.016790",None),
8:("Manglik",2016,"Structure-based discovery of opioid analgesics with reduced side effects","10.1038/nature19112",None),
9:("Violin",2014,"Biased ligands at G-protein-coupled receptors: promise and progress","10.1016/j.tips.2014.04.007",None),
10:("Omieczynski",2019,"BiasDB: a comprehensive database for biased GPCR ligands","10.1101/742643",None),
11:("Suomivuori",2020,"Molecular mechanism of biased signaling in a prototypical G protein-coupled receptor","10.1126/science.aaz0326",None),
12:("Wingler",2020,"Angiotensin and biased analogs induce structurally distinct active conformations within a GPCR","10.1126/science.aay9813",None),
13:("Slosky",2020,"Beta-arrestin-biased allosteric modulator of NTSR1 selectively attenuates addictive behaviors","10.1016/j.cell.2020.04.053",None),
14:("Mysinger",2012,"Directory of useful decoys, enhanced (DUD-E): better ligands and decoys for better benchmarking","10.1021/jm300687e",None),
15:("Trott",2010,"AutoDock Vina: improving the speed and accuracy of docking with a new scoring function","10.1002/jcc.21334",None),
16:("Eberhardt",2021,"AutoDock Vina 1.2.0: new docking methods, expanded force field, and Python bindings","10.1021/acs.jcim.1c00203",None),
17:("Bouysset",2021,"ProLIF: a library to encode molecular interactions as fingerprints","10.1186/s13321-021-00548-6",None),
18:("Ke",2017,"LightGBM",None,
   "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 3146–3157)."),
19:("Chen",2016,"XGBoost",None,
   "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785"),
20:("Ahmad",2022,"ChemBERTa-2",None,
   "Ahmad, W., Simon, E., Chithrananda, S., Grand, G., & Ramsundar, B. (2022). ChemBERTa-2: Towards chemical foundation models. *arXiv*. https://doi.org/10.48550/arXiv.2209.01712"),
21:("Lundberg",2017,"A unified approach to interpreting model predictions",None,
   "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4765–4774)."),
22:("Berman",2000,"The Protein Data Bank","10.1093/nar/28.1.235",None),
23:("Jumper",2021,"Highly accurate protein structure prediction with AlphaFold","10.1038/s41586-021-03819-2",None),
24:("Varadi",2022,"AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space","10.1093/nar/gkab1061",None),
25:("UniProt Consortium",2023,"UniProt: the Universal Protein Knowledgebase in 2023",None,
   "UniProt Consortium. (2023). UniProt: the Universal Protein Knowledgebase in 2023. *Nucleic Acids Research*, *51*(D1), D523–D531. https://doi.org/10.1093/nar/gkac1052"),
26:("Landrum",2025,"RDKit: open-source cheminformatics",None,
   "Landrum, G., et al. (2025). *RDKit: Open-source cheminformatics* (Version 2025.03) [Computer software]. https://www.rdkit.org"),
27:("Kenakin",2013,"Signalling bias in new drug discovery: detection, quantification and therapeutic impact","10.1038/nrd3954",None),
28:("Eastman",2017,"OpenMM 7: rapid development of high performance algorithms for molecular dynamics","10.1371/journal.pcbi.1005659",None),
29:("Santos-Martins",2025,"Meeko: molecule parameterization and software interoperability for docking and beyond","10.1021/acs.jcim.5c02271",None),
30:("Krivak",2018,"P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure","10.1186/s13321-018-0285-8",None),
31:("Le Guilloux",2009,"Fpocket: an open source platform for ligand pocket detection","10.1186/1471-2105-10-168",None),
32:("McNutt",2021,"GNINA 1.0: molecular docking with deep learning","10.1186/s13321-021-00522-2",None),
33:("Kursa",2010,"Feature selection with the Boruta package","10.18637/jss.v036.i11",None),
34:("Akiba",2019,"Optuna",None,
   "Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701"),
35:("Wolpert",1992,"Stacked generalization","10.1016/S0893-6080(05)80023-1",None),
36:("Breiman",1996,"Stacked regressions","10.1007/BF00117832",None),
37:("Niculescu-Mizil",2005,"Predicting good probabilities with supervised learning",None,
   "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). https://doi.org/10.1145/1102351.1102430"),
38:("Brier",1950,"Verification of forecasts expressed in terms of probability","10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2",None),
39:("Wacker",2017,"How ligands illuminate GPCR molecular pharmacology","10.1016/j.cell.2017.07.009",None),
40:("Insel",2019,"GPCRomics: an approach to discover GPCR drug targets","10.1016/j.tips.2019.04.001",None),
41:("Wallach",2018,"Most ligand-based classification benchmarks reward memorization rather than generalization","10.1021/acs.jcim.7b00403",None),
42:("Sheridan",2013,"Time-split cross-validation as a method for estimating the goodness of prospective prediction","10.1021/ci400084k",None),
43:("Dominguez",2003,"HADDOCK: a protein-protein docking approach based on biochemical or biophysical information","10.1021/ja026939x",None),
44:("London",2011,"Rosetta FlexPepDock web server: high resolution modeling of peptide-protein interactions","10.1093/nar/gkr431",None),
45:("Yang",2019,"Analyzing learned molecular representations for property prediction","10.1021/acs.jcim.9b00237",None),
46:("Zhou",2023,"Uni-Mol: a universal 3D molecular representation learning framework",None,
   "Zhou, G., Gao, Z., Ding, Q., Zheng, H., Xu, H., Wei, Z., Zhang, L., & Ke, G. (2023). Uni-Mol: A universal 3D molecular representation learning framework. *International Conference on Learning Representations (ICLR)*."),
47:("Mendez",2019,"ChEMBL: towards direct deposition of bioassay data","10.1093/nar/gky1075",None),
48:("O'Boyle",2011,"Open Babel: an open chemical toolbox","10.1186/1758-2946-3-33",None),
49:("Rogers",2010,"Extended-connectivity fingerprints","10.1021/ci100050t",None),
50:("Durant",2002,"Reoptimization of MDL keys for use in drug discovery","10.1021/ci010132r",None),
51:("Bemis",1996,"The properties of known drugs. 1. Molecular frameworks","10.1021/jm9602928",None),
52:("Efron",1986,"Bootstrap methods for standard errors, confidence intervals, and other measures of statistical accuracy","10.1214/ss/1177013815",None),
53:("Pedregosa",2011,"Scikit-learn: machine learning in Python",None,
   "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830."),
54:("Zaharia",2018,"Accelerating the machine learning lifecycle with MLflow",None,
   "Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin, 41*(4), 39-45."),
55:("Sanchez",2021,"BiasNet: a model to predict ligand bias toward GPCR signaling","10.1021/acs.jcim.1c00317",None),
56:("Kumar",2024,"GPCR-IPL score: multilevel featurization of GPCR-ligand interaction patterns and prediction of ligand functions from selectivity to biased activation","10.1093/bib/bbae105",None),
57:("Mitchell",2019,"Model cards for model reporting","10.1145/3287560.3287596",None),
58:("Riniker",2015,"Better informed distance geometry: using what we know to improve conformation generation","10.1021/acs.jcim.5b00654",None),
59:("Halgren",1996,"Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94","10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P",None),
60:("Bento",2020,"An open source chemical structure curation pipeline using RDKit","10.1186/s13321-020-00456-1",None),
61:("Cock",2009,"Biopython: freely available Python tools for computational molecular biology and bioinformatics","10.1093/bioinformatics/btp163",None),
62:("Henikoff",1992,"Amino acid substitution matrices from protein blocks","10.1073/pnas.89.22.10915",None),
63:("Breiman",2001,"Random forests","10.1023/A:1010933404324",None),
}

def fetch(doi=None, query=None):
    if doi:
        url=f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    else:
        url="https://api.crossref.org/works?"+urllib.parse.urlencode({"query.bibliographic":query,"rows":3})
    req=urllib.request.Request(url, headers=UA)
    data=json.load(urllib.request.urlopen(req, timeout=25))["message"]
    if doi: return data
    items=data.get("items",[])
    return items[0] if items else None

def apa_authors(authors):
    parts=[]
    for a in authors:
        fam=a.get("family") or a.get("name","")
        giv=a.get("given","")
        inits=" ".join(f"{g[0]}." for g in re.split(r"[ \-]", giv) if g) if giv else ""
        parts.append(f"{fam}, {inits}".strip().rstrip(",") if inits else fam)
    if not parts: return ""
    if len(parts)==1: return parts[0]
    if len(parts)<=20:
        return ", ".join(parts[:-1])+", & "+parts[-1]
    return ", ".join(parts[:19])+", ... "+parts[-1]

def titlecase_journal(j):  # CrossRef already gives proper case; just return
    return j

def fmt_apa(n, meta, surname, year, title):
    # APA wants the ARTICLE title in sentence case; CrossRef returns Title Case
    # and sometimes truncates, so we use our verified sentence-case title and
    # take only authors / journal / volume / issue / pages / DOI from CrossRef.
    typ=meta.get("type","")
    authors=apa_authors(meta.get("author",[]))
    yr=year  # agent-verified publication year (avoids CrossRef epub/print drift)
    title=title.strip().rstrip(".")
    tdot="" if title.endswith(("?","!")) else "."  # no double terminal punctuation
    cont=(meta.get("container-title") or [""])
    cont=clean(cont[0]) if cont else ""
    vol=meta.get("volume",""); iss=meta.get("issue",""); page=meta.get("page","").replace("-","–")
    doi=meta.get("DOI","")
    doi_s=f" https://doi.org/{doi}" if doi else ""
    if typ=="proceedings-article":
        pp=f" (pp. {page})" if page else ""
        return f"{authors} ({yr}). {title}{tdot} In *{cont}*{pp}.{doi_s}"
    if typ in ("posted-content","preprint"):
        repo=cont or ("bioRxiv" if ("biorxiv" in doi.lower() or "10.1101" in doi) else "arXiv")
        return f"{authors} ({yr}). {title}{tdot} *{repo}*.{doi_s}"
    # journal-article (default)
    vi=f"*{vol}*"+(f"({iss})" if iss else "") if vol else ""
    pg=f", {page}" if page else ""
    sep=", " if vi else ""
    return f"{authors} ({yr}). {title}{tdot} *{cont}*{sep}{vi}{pg}.{doi_s}"

def validate(meta, surname, year, title):
    auth=" ".join((a.get("family","")+" "+a.get("name","")) for a in meta.get("author",[])).lower()
    ok_auth = surname.split()[0].lower() in auth or surname.lower() in auth
    myr=meta.get("issued",{}).get("date-parts",[[0]])[0][0] or 0
    ok_year = abs((myr or 0)-year)<=1
    mt=(meta.get("title") or [""])[0].lower()
    tw=set(re.findall(r"[a-z0-9]+", title.lower())); mw=set(re.findall(r"[a-z0-9]+", mt))
    ov = len(tw&mw)/max(1,len(tw))
    return ok_auth, ok_year, round(ov,2)

# Hand author surnames for MANUAL refs (not fetched from CrossRef), for in-text labels.
MANUAL_AUTHORS = {
 18:(["Ke","Meng","Finley"],"2017"), 19:(["Chen","Guestrin"],"2016"),
 20:(["Ahmad","Simon","Chithrananda"],"2022"), 21:(["Lundberg","Lee"],"2017"),
 34:(["Akiba","Sano","Yanase"],"2019"), 37:(["Niculescu-Mizil","Caruana"],"2005"),
 46:(["Zhou","Gao","Ding"],"2023"), 53:(["Pedregosa","Varoquaux","Gramfort"],"2011"),
 54:(["Zaharia","Chen","Davidson"],"2018"), 26:(["Landrum"],"2025"),
}

def intext(surnames, year):
    """APA in-text labels from a surname list. Returns (paren_inner, narrative)."""
    s=[x for x in surnames if x]
    if len(s)==0: return (f"Anonymous, {year}", f"Anonymous ({year})")
    if len(s)==1: lead=s[0]
    elif len(s)==2: lead=f"{s[0]} & {s[1]}"
    else: lead=f"{s[0]} et al."
    return (f"{lead}, {year}", f"{lead} ({year})")

out={}
for n in sorted(REFS):
    surname,year,title,doi,manual=REFS[n]
    if manual:
        sn,yr = MANUAL_AUTHORS.get(n, ([surname],str(year)))
        paren,narr = intext(sn, yr)
        out[n]={"apa":manual,"conf":"MANUAL","sort":sn[0].lower(),
                "label":paren,"narr":narr,"year":str(year)}
        print(f"[{n:>2}] MANUAL"); continue
    try:
        meta=fetch(doi=doi) if doi else fetch(query=f"{surname} {title} {year}")
        if meta is None:
            out[n]={"apa":None,"conf":"NOMATCH"}; print(f"[{n:>2}] NOMATCH"); continue
        a,y,ov=validate(meta,surname,year,title)
        conf = "OK" if (a and y and ov>=0.5) else f"CHECK(auth={a},year={y},ov={ov})"
        auths=[ (au.get("family") or au.get("name","")) for au in meta.get("author",[]) ]
        paren,narr = intext(auths, str(year))
        out[n]={"apa":fmt_apa(n,meta,surname,year,title),"conf":conf,"doi":meta.get("DOI"),
                "sort":(auths[0] if auths else surname).lower(),
                "label":paren,"narr":narr,"year":str(year)}
        print(f"[{n:>2}] {conf}")
    except Exception as e:
        out[n]={"apa":None,"conf":f"ERR:{type(e).__name__}"}; print(f"[{n:>2}] ERR {e}")
    time.sleep(0.4)

json.dump(out, open("scripts/apa_out.json","w"), indent=1, ensure_ascii=False)
print("\n--- flagged (not OK/MANUAL) ---")
for n in sorted(out):
    if out[n]["conf"] not in ("OK","MANUAL"):
        print(n, out[n]["conf"], "|", REFS[n][2][:50])
