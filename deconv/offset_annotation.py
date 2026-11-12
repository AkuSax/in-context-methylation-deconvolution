"""Does the sorted-array vs bulk-array per-CpG offset follow probe annotation?

Post hoc diagnostic (seventh development look at GSE110554; uses its truth labels).
Question: the v5 clean penalty is false `other` driven by a per-CpG offset between sorted
arrays and mixture arrays. If that offset is predictable from probe properties that are
known in advance (design type, colour channel, island context, SNP/cross-reactive masks),
those properties can be given to the model as CpG features and used to shape the
simulator's offsets.

Parts:
1. GSE110554 (Noob): offset d = mixture - reference x truth at every complete CpG, beta
   and M scale. Split-half reliability with disjoint sorted arrays and disjoint mixtures.
2. Association of |d| and d with annotation, with and without the quantities the model
   already sees (reference mean, reference SD, between-type range). Cross-validated by
   chromosome parity.
3. The primary 25-per-side marker panel: annotation of the CpGs with recurrent large
   standardised residuals.
4. Label-free replication on a training source: GSE35069 (450K, deposited) whole blood
   against its own sorted fractions, fractions fitted by NNLS. Correlate its offset with
   GSE110554's at shared CpGs.

Run from deconv/:  DECONV_PREPROC=noob python offset_annotation.py
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("DECONV_PREPROC", "noob")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

import datasets as ds
import reference
import simulator

OUT = Path(__file__).resolve().parent / "results" / "offset_annotation_gse110554"
R3 = Path(os.environ.get("DECONV_DATA", str(Path.home() / "data" / "comp-genomics"))) / "discovery" / "r3" / "raw"
MANIFEST = R3 / "gpl21145_epicv1_manifest" / "GPL21145_MethylationEPIC_15073387_v-1-0.csv.gz"
ZHOU = R3 / "zhou_epic_masks" / "EPIC.hg38.mask.tsv.gz"
EPS = 1e-3


def logit2(v):
    v = np.clip(v, EPS, 1 - EPS)
    return np.log2(v / (1 - v))


def load_annotation() -> pd.DataFrame:
    with gzip.open(MANIFEST, "rt") as f:
        skip = next(i for i, line in enumerate(f) if line.startswith("IlmnID"))
    cols = ["IlmnID", "Infinium_Design_Type", "Color_Channel", "CHR",
            "Relation_to_UCSC_CpG_Island", "SNP_ID", "Methyl450_Loci"]
    m = pd.read_csv(MANIFEST, skiprows=skip, usecols=cols, low_memory=False, dtype=str)
    m = m[m.IlmnID.str.startswith("cg", na=False)].set_index("IlmnID")
    z = pd.read_csv(ZHOU, sep="\t", dtype=str).set_index("Probe_ID")
    a = pd.DataFrame(index=m.index)
    a["design"] = m.Infinium_Design_Type
    a["channel"] = np.where(m.Infinium_Design_Type == "I", "I-" + m.Color_Channel.fillna("?"), "II")
    a["island"] = m.Relation_to_UCSC_CpG_Island.fillna("OpenSea").str.replace("^[NS]_", "", regex=True)
    a["snp_manifest"] = m.SNP_ID.notna()
    a["in_450k"] = m.Methyl450_Loci.eq("TRUE") | m.Methyl450_Loci.eq("True")
    a["chr"] = m.CHR
    zz = z.reindex(a.index)
    a["zhou_general"] = zz.M_general.eq("TRUE")
    a["zhou_any"] = zz.maskUniq.notna()
    return a


def offset_table(x_sorted, lab, x_mix, truth, types):
    """Reference x truth residual at every row. Returns mean d (beta), mean dM, ref stats."""
    ref = np.stack([x_sorted[:, lab == t].mean(1) for t in types], 1)
    w = truth[types].to_numpy().T
    w = w / w.sum(0, keepdims=True)
    pred = ref @ w
    return x_mix - pred, logit2(x_mix) - logit2(pred), ref


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    ann = load_annotation()
    d = ds.load("GSE110554")
    srt = d.cell_type.dropna()
    mix = d.truth.dropna(how="all").index
    types = ["Neu", "Mono", "B", "CD4T", "CD8T", "NK"]
    b = d.betas
    keep = reference.usable_probes(b[srt.index], b[mix]).intersection(ann.index)
    keep = keep[~ann.loc[keep, "chr"].isin(["X", "Y"])]
    xs = b.loc[keep, srt.index].to_numpy(float)
    lab = srt.to_numpy()
    xm = b.loc[keep, mix].to_numpy(float)
    tr = d.truth.loc[mix]
    say(f"GSE110554 Noob: {len(keep):,} autosomal complete CpGs, {len(srt)} sorted, {len(mix)} mixtures")

    D, DM, ref = offset_table(xs, lab, xm, tr, types)
    off, offM = D.mean(1), DM.mean(1)
    say(f"offset beta: mean {off.mean():+.4f}, SD {off.std():.4f}, share positive {np.mean(off > 0):.2f}")
    say(f"offset M: median {np.median(offM):+.3f}, SD {offM.std():.3f} log2")

    # split-half reliability: disjoint sorted arrays (alternate within type) and mixtures
    rng = np.random.default_rng(0)
    half = np.zeros(len(lab), bool)
    for t in types:
        idx = rng.permutation(np.flatnonzero(lab == t))
        half[idx[: len(idx) // 2]] = True
    mperm = rng.permutation(len(mix))
    m1, m2 = np.sort(mperm[:6]), np.sort(mperm[6:])
    _, A, _ = offset_table(xs[:, half], lab[half], xm[:, m1], tr.iloc[m1], types)
    _, B, _ = offset_table(xs[:, ~half], lab[~half], xm[:, m2], tr.iloc[m2], types)
    a1, b1 = A.mean(1), B.mean(1)
    r_half = np.corrcoef(a1, b1)[0, 1]
    r_abs = stats.spearmanr(np.abs(a1), np.abs(b1)).statistic
    say(f"split-half (disjoint sorted arrays and mixtures), M offset: Pearson {r_half:.3f}, "
        f"Spearman of |offset| {r_abs:.3f}")
    # a pure-noise check: shuffle sorted labels within halves gives the noise floor? use
    # the same split but swap which reference half pairs with which mixture half
    _, A2, _ = offset_table(xs[:, ~half], lab[~half], xm[:, m1], tr.iloc[m1], types)
    say(f"  same mixtures, other reference half: Pearson {np.corrcoef(a1, A2.mean(1))[0, 1]:.3f} "
        "(shared mixtures; upper bound)")

    # --- 2. association with annotation -----------------------------------------------
    f = ann.loc[keep].copy()
    f["ref_mean"] = ref.mean(1)
    f["ref_range"] = ref.max(1) - ref.min(1)
    sd_within = np.sqrt(np.mean([xs[:, lab == t].var(1, ddof=1) for t in types], 0))
    f["ref_sd"] = sd_within
    f["bulk_mean"] = xm.mean(1)
    f["off"], f["offM"], f["absM"] = off, offM, np.abs(offM)
    f["beta_bin"] = pd.cut(f.bulk_mean, [0, .1, .3, .7, .9, 1], include_lowest=True).astype(str)
    say("\n|M offset| and signed offset by annotation (GSE110554):")
    rows = []
    for col in ["design", "channel", "island", "snp_manifest", "zhou_general", "zhou_any", "in_450k", "beta_bin"]:
        g = f.groupby(col)
        t = pd.DataFrame({"n": g.size(), "median |dM|": g.absM.median(), "mean dM": g.offM.mean(),
                          "mean d beta": g.off.mean(), "share dM>0": g.offM.apply(lambda v: (v > 0).mean()),
                          "P(|dM| top 5%)": g.absM.apply(lambda v: (v > f.absM.quantile(.95)).mean())})
        t.insert(0, "feature", col)
        rows.append(t.reset_index().rename(columns={col: "level"}))
    by_ann = pd.concat(rows, ignore_index=True)
    by_ann["level"] = by_ann["level"].astype(str)
    by_ann.to_csv(OUT / "offset_by_annotation.csv", index=False)
    say(by_ann.round(3).to_string(index=False))

    # stratified: design/channel within mid-beta only, to separate from beta level
    mid = f.bulk_mean.between(.3, .7)
    say("\nwithin mid-beta CpGs (bulk 0.3-0.7) only:")
    say(f.loc[mid].groupby("channel").agg(n=("absM", "size"), med_abs=("absM", "median"),
                                           mean_dM=("offM", "mean")).round(3).to_string())

    # cross-validated R^2: visible-only vs visible + annotation
    def design_matrix(df, with_ann):
        bm = df.bulk_mean.clip(EPS, 1 - EPS)
        X = pd.DataFrame({"mid": bm * (1 - bm), "mid2": (bm * (1 - bm)) ** 2, "bm": bm,
                          "range": df.ref_range, "sd": df.ref_sd, "sd2": df.ref_sd ** 2})
        if with_ann:
            X = X.join(pd.get_dummies(df[["channel", "island"]], drop_first=True).astype(float))
            X[["snp", "zg", "za", "k450"]] = df[["snp_manifest", "zhou_general", "zhou_any", "in_450k"]].astype(float).to_numpy()
            for c in [c for c in X.columns if c.startswith("channel_")]:
                X[c + "*mid"] = X[c] * X["mid"]
        X.insert(0, "one", 1.0)
        return X.to_numpy(float), list(X.columns)

    odd = pd.to_numeric(f.chr, errors="coerce").fillna(0).astype(int) % 2 == 1
    res = {}
    for target in ["absM", "offM"]:
        y = f[target].to_numpy()
        for with_ann in (False, True):
            X, names = design_matrix(f, with_ann)
            pred = np.empty_like(y)
            for tr_mask in (odd.to_numpy(), ~odd.to_numpy()):
                coef, *_ = np.linalg.lstsq(X[tr_mask], y[tr_mask], rcond=None)
                pred[~tr_mask] = X[~tr_mask] @ coef
            r2 = 1 - np.mean((y - pred) ** 2) / np.var(y)
            top = f.absM.to_numpy() > np.quantile(f.absM, .95)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(top, pred if target == "absM" else np.abs(pred))
            res[f"{target}_{'ann' if with_ann else 'visible'}"] = dict(r2=round(r2, 4), auroc_top5=round(auc, 4))
    # ceiling: how much of |dM| replicates across halves
    res["split_half_spearman_abs"] = round(float(r_abs), 4)
    res["split_half_pearson_signed"] = round(float(r_half), 4)
    say("\nchromosome-parity cross-validated fit (R^2, AUROC for top-5% |dM|):")
    say(json.dumps(res, indent=1))

    # --- 3. primary marker panel ------------------------------------------------------
    mk = reference.pick_markers(b.loc[keep, srt.index], srt, n_per_side=25)
    pos = keep.get_indexer(mk)
    R = D[pos]
    z = R / (1.4826 * np.median(np.abs(R - np.median(R, 0)), 0))
    freq = (np.abs(z) > 3).sum(1)
    fm = f.iloc[pos].assign(n_mix_z3=freq, mean_abs_z=np.abs(z).mean(1))
    rec = fm.n_mix_z3 >= 3
    say(f"\nprimary panel {len(mk)} CpGs; recurrent (|z|>3 in >=3 of 12 mixtures): {int(rec.sum())}; "
        f"they hold {freq[rec.to_numpy()].sum()} of {freq.sum()} |z|>3 cells")
    comp = []
    for col in ["channel", "island", "snp_manifest", "zhou_any", "beta_bin"]:
        ct = pd.crosstab(fm[col], rec)
        comp.append(f"{col}:\n{ct.to_string()}")
    say("\n".join(comp))
    for col in ["snp_manifest", "zhou_any"]:
        ct = pd.crosstab(fm[col], rec).reindex(index=[False, True], columns=[False, True], fill_value=0)
        say(f"Fisher {col}: OR {stats.fisher_exact(ct.to_numpy())[0]:.2f}, p {stats.fisher_exact(ct.to_numpy())[1]:.3g}")
    ct = pd.crosstab(fm.channel != "II", rec).reindex(index=[False, True], columns=[False, True], fill_value=0)
    say(f"Fisher type I vs II: OR {stats.fisher_exact(ct.to_numpy())[0]:.2f}, p {stats.fisher_exact(ct.to_numpy())[1]:.3g}")
    fm.drop(columns=["chr"]).to_csv(OUT / "panel_markers.csv")

    # --- 4. label-free replication: GSE35069 whole blood vs own sorted -------------------
    meta = simulator.read_meta(simulator.DEFAULT_GSE35069)
    allb = simulator.read_betas(simulator.DEFAULT_GSE35069, meta)
    typ = pd.Series([t.rsplit("_", 1)[0] for t in meta.title], index=list(meta.gsm))
    sorted35 = typ[typ.isin(simulator.SORTED_TYPES)]
    bulk35 = typ.index[typ.isin(["WB", "PBMC"])]
    c35 = reference.usable_probes(allb[sorted35.index], allb[bulk35]).intersection(ann.index)
    c35 = c35[~ann.loc[c35, "chr"].isin(["X", "Y"])]
    S = allb.loc[c35, sorted35.index].to_numpy(float)
    t35 = list(pd.unique(sorted35))
    lab35 = sorted35.to_numpy()
    ref35 = np.stack([S[:, lab35 == t].mean(1) for t in t35], 1)
    mk35 = c35.get_indexer(reference.pick_markers(allb.loc[c35, sorted35.index], sorted35, 25))
    Xb = allb.loc[c35, bulk35].to_numpy(float)
    W = np.stack([nnls(ref35[mk35], Xb[mk35, j])[0] for j in range(Xb.shape[1])], 1)
    W = W / W.sum(0, keepdims=True)
    P = ref35 @ W
    off35M = (logit2(Xb) - logit2(P)).mean(1)
    off35 = (Xb - P).mean(1)
    say(f"\nGSE35069 (450K, deposited; label-free NNLS fit on {len(mk35)} markers): {len(c35):,} CpGs, "
        f"{len(bulk35)} bulk arrays (6 WB + 6 PBMC)")
    say(f"offset beta: mean {off35.mean():+.4f}, SD {off35.std():.4f}; M SD {off35M.std():.3f}")
    f35 = ann.loc[c35].assign(absM=np.abs(off35M), offM=off35M, bulk_mean=Xb.mean(1))
    say(f35.groupby("channel").agg(n=("absM", "size"), med_abs=("absM", "median"),
                                   mean_dM=("offM", "mean")).round(3).to_string())
    say(f35.groupby("island").agg(n=("absM", "size"), med_abs=("absM", "median"),
                                  mean_dM=("offM", "mean")).round(3).to_string())
    shared = c35.intersection(keep)
    s110 = pd.Series(offM, index=keep).loc[shared]
    s35 = pd.Series(off35M, index=c35).loc[shared]
    say(f"shared CpGs {len(shared):,}: Pearson of signed M offsets {np.corrcoef(s110, s35)[0, 1]:.3f}; "
        f"Spearman of |offset| {stats.spearmanr(s110.abs(), s35.abs()).statistic:.3f}")
    # partial out beta level: rank-correlate residuals of |offset| after binning by bulk beta
    bins = pd.qcut(f.loc[shared, "bulk_mean"], 20, labels=False, duplicates="drop")
    ra = s110.abs() - s110.abs().groupby(bins).transform("median")
    rb = s35.abs() - s35.abs().groupby(bins).transform("median")
    say(f"  same, within 20 bulk-beta bins: Spearman {stats.spearmanr(ra, rb).statistic:.3f}")
    recI = fm.index[rec.to_numpy()].intersection(c35)
    say(f"recurrent GSE110554 panel CpGs also on 450K: {len(recI)}; their median |GSE35069 offset| "
        f"{s35.abs().reindex(recI).median():.3f} vs all shared {s35.abs().median():.3f} "
        f"(percentile {stats.percentileofscore(s35.abs(), s35.abs().reindex(recI).median()):.0f})")

    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    (OUT / "run.log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
