"""Post hoc deployment-risk audit of saved DEVELOPMENT predictions; no model runs.

Usage (repository root): python deconv/audit_deployment_tradeoff.py
Uses only Python's standard library. Never opens reserved evaluation data.
The mixture weight is hypothetical, not an estimated prevalence of missing types.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def mse(rows):
    """Equal weight per sample/scenario, then per retained type within that case."""
    if not rows:
        raise ValueError("Empty evaluation stratum")
    return 10000 * sum(float(r["sq_err_sum"]) / int(r["n_cells"]) for r in rows) / len(rows)


def identifiability_example():
    # A and B are known signatures. An unmeasured C lies in their convex hull.
    a, b = [0.1, 0.9, 0.2], [0.9, 0.1, 0.8]
    c = [(x + y) / 2 for x, y in zip(a, b)]
    complete = [0.5 * x + 0.5 * y for x, y in zip(a, b)]
    incomplete = [0.4 * x + 0.4 * y + 0.2 * z for x, y, z in zip(a, b, c)]
    difference = max(abs(x - y) for x, y in zip(complete, incomplete))
    if difference > 1e-12:
        raise AssertionError("Counterexample does not match")
    return dict(reference_A=a, reference_B=b, unknown_C=c,
                world_complete=dict(A=0.5, B=0.5, other=0.0),
                world_incomplete=dict(A=0.4, B=0.4, other=0.2),
                observed_bulk=complete, max_absolute_difference=difference)


def audit(source):
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"per_side", "dropped", "method", "sample", "sq_err_sum", "n_cells",
                "other_est", "other_true"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Expected eval_real_dev.py estimates_long.csv schema")
    keys = [(r["per_side"], r["dropped"], r["method"], r["sample"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate predictions")
    for row in rows:
        if int(row["n_cells"]) < 1 or not math.isfinite(float(row["sq_err_sum"])) or float(row["sq_err_sum"]) < 0:
            raise ValueError("Invalid squared errors or type count")
        if not all(math.isfinite(float(row[k])) for k in ("other_est", "other_true")):
            raise ValueError("Non-finite other estimate or truth")

    summaries, curves = [], []
    for panel in sorted({int(r["per_side"]) for r in rows}):
        part = [r for r in rows if int(r["per_side"]) == panel]
        for scenario, include in (
            ("all_drops", lambda r: r["dropped"] != "none"),
            ("without_neutrophils", lambda r: r["dropped"] not in ("none", "Neu")),
            ("abundant_without_neutrophils", lambda r: r["dropped"] not in ("none", "Neu") and float(r["other_true"]) >= 0.05),
        ):
            selected = {m: dict(clean=[r for r in part if r["method"] == m and r["dropped"] == "none"],
                                missing=[r for r in part if r["method"] == m and include(r)])
                        for m in ("model", "RPC")}
            for regime in ("clean", "missing"):
                left, right = selected["model"][regime], selected["RPC"][regime]
                key = lambda r: (r["sample"], r["dropped"], r["n_cells"], r["other_true"])
                if sorted(map(key, left)) != sorted(map(key, right)):
                    raise ValueError("Unmatched model/RPC rows")
            losses = {m: {regime: mse(s) for regime, s in v.items()} for m, v in selected.items()}
            clean_cost = losses["model"]["clean"] - losses["RPC"]["clean"]
            missing_benefit = losses["RPC"]["missing"] - losses["model"]["missing"]
            crossing = clean_cost / (clean_cost + missing_benefit) if clean_cost + missing_benefit else None
            report = dict(per_side=panel, scenario=scenario,
                          n_clean=len(selected["model"]["clean"]),
                          n_missing_cases=len(selected["model"]["missing"]),
                          n_unique_mixtures=len({r["sample"] for r in selected["model"]["missing"]}),
                          model_clean_rmse=math.sqrt(losses["model"]["clean"]),
                          rpc_clean_rmse=math.sqrt(losses["RPC"]["clean"]),
                          model_missing_rmse=math.sqrt(losses["model"]["missing"]),
                          rpc_missing_rmse=math.sqrt(losses["RPC"]["missing"]),
                          crossing_prevalence=crossing,
                          crossing_within_zero_one=crossing is not None and 0 <= crossing <= 1,
                          model_wins_when_prevalence_above_crossing=clean_cost + missing_benefit > 0)
            # This existing 5-point threshold is descriptive; it is not selected or recommended here.
            for regime in ("clean", "missing"):
                rr = selected["model"][regime]
                report[regime + "_fraction_other_above_5pts"] = sum(float(r["other_est"]) > 0.05 for r in rr) / len(rr)
            summaries.append(report)
            for prevalence in (0.0, 0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
                risks = {m: math.sqrt((1 - prevalence) * loss["clean"] + prevalence * loss["missing"])
                         for m, loss in losses.items()}
                curves.append(dict(per_side=panel, scenario=scenario, hypothetical_missing_prevalence=prevalence,
                                   model_rmse_pts=risks["model"], rpc_rmse_pts=risks["RPC"],
                                   model_minus_rpc_pts=risks["model"] - risks["RPC"]))
    return summaries, curves


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT / "results/dev_gse110554_gpu_v5/estimates_long.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results/research_audit_20261112")
    args = ap.parse_args()
    summaries, curves = audit(args.source)
    output = dict(status="post_hoc_development_only", source=str(args.source.resolve()),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  loss="mean over cases of mean squared full-mixture fraction error over kept types, in points squared",
                  formula="risk(p) = sqrt((1-p)*MSE_clean + p*MSE_missing)",
                  limitations=["Hypothetical reweighting of one repeatedly examined development study, not deployment validation.",
                               "Missing scenarios repeat the same 12 mixtures; they are not independent observations.",
                               "Within each missing stratum, cases retain their observed equal weights; these are not measured clinical frequencies.",
                               "No uncertainty interval, calibrated classifier, or real prevalence estimate is supplied.",
                               "An other estimate above 5 points is an existing descriptive threshold, not a validated decision rule."],
                  identifiability_counterexample=identifiability_example(), summaries=summaries)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.out / "mixed_risk.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    for row in summaries:
        if row["per_side"] == 25:
            print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
