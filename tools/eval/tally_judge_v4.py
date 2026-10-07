"""Tally protocol-v4 verdicts (tools/eval/JUDGE_V4.md): 3 judges, majority vote.

    python tools/eval/tally_judge_v4.py tools/eval/out/judge-v4/<name> [--pairs v4-4b:v3-4b v4-2b:v3-2b]
                                        [--sets eval4] [--boot 10000]

Reads KEY.json + verdicts/<packet>.jsonl (one judge output per packet file). For every
(row, run): each judge's grade of the letter holding that run's output; majority over judges for
pass / harmful / acted, plurality for fail_type. Writes summary.json and summary.md next to KEY:

* per run: pass rate, harmful-failure rate, acted rate (overall, per set, per category);
* inter-judge agreement over the distinct outputs: unanimous share, mean pairwise agreement and
  Fleiss' kappa for pass, and for harmful;
* paired comparisons: rows only A passes / only B passes, pass-rate and harmful-rate differences
  with paired bootstrap 95% CIs (rows resampled with replacement);
* failure types per run (majority-failed outputs).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def read_jsonl(p: Path) -> list[dict]:
    out = []
    for x in p.read_text(encoding="utf-8-sig").splitlines():
        x = x.strip()
        if x:
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:
                pass
    return out


def fleiss(items: list[list[int]]) -> float:
    """items: per item, the votes (0/1) of every rater (same count each)."""
    items = [v for v in items if len(v) >= 2]
    if not items:
        return float("nan")
    n = len(items[0])
    items = [v for v in items if len(v) == n]
    N = len(items)
    p1 = sum(sum(v) for v in items) / (N * n)
    pe = p1 ** 2 + (1 - p1) ** 2
    P = [(sum(v) * (sum(v) - 1) + (n - sum(v)) * (n - sum(v) - 1)) / (n * (n - 1)) for v in items]
    po = sum(P) / N
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def boot_ci(diffs: list[float], B: int, seed: int = 4) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    if not n:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return sum(diffs) / n, means[int(0.025 * B)], means[int(0.975 * B) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B run pairs for paired comparisons")
    ap.add_argument("--sets", nargs="*", help="restrict to these sets (default all)")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    K = json.loads((a.dir / "KEY.json").read_text(encoding="utf-8-sig"))
    runs, nj = K["runs"], K["judges"]
    # grades[(key, judge)][letter] = grade
    grades: dict[tuple[str, str], dict[str, dict]] = {}
    for f in sorted((a.dir / "verdicts").glob("*.jsonl")):
        j = f.stem.split("-")[0].lstrip("j")
        for v in read_jsonl(f):
            grades[(v.get("key"), j)] = {g.get("letter"): g for g in v.get("grades", [])}
    rows = {k: v for k, v in K["rows"].items() if not a.sets or v["set"] in a.sets}
    res: dict[str, dict[str, dict]] = {run: {} for run in runs}   # run -> key -> majority record
    agree_pass, agree_harm, incomplete = [], [], []
    for k, info in rows.items():
        per_text: dict[str, list[dict]] = collections.defaultdict(list)   # one entry per distinct output
        for j, letters in info["judges"].items():
            g = grades.get((k, j))
            for L, rs in letters.items():
                gg = (g or {}).get(L)
                if gg is None:
                    continue
                per_text[",".join(sorted(rs))].append(gg)
        for grp, gs in per_text.items():
            if len(gs) < nj:
                incomplete.append((k, grp, len(gs)))
            if len(gs) >= 2:
                agree_pass.append([int(bool(g.get("pass"))) for g in gs])
                agree_harm.append([int(bool(g.get("harmful"))) for g in gs])
            if not gs:
                continue
            npass = sum(bool(g.get("pass")) for g in gs)
            maj_pass = npass * 2 > len(gs)
            harm = sum(bool(g.get("harmful")) for g in gs) * 2 > len(gs)
            acted = sum(bool(g.get("acted")) for g in gs) * 2 > len(gs)
            ft = collections.Counter(g.get("fail_type") or "other" for g in gs if not g.get("pass"))
            rec = {"pass": maj_pass, "harmful": harm and not maj_pass, "acted": acted, "votes": npass, "n": len(gs),
                   "fail_type": ft.most_common(1)[0][0] if (ft and not maj_pass) else "",
                   "reasons": [g.get("reason", "") for g in gs if not g.get("pass")],
                   "cat": info["cat"], "set": info["set"]}
            for run in grp.split(","):
                res[run][k] = rec
    keys = sorted(set.intersection(*(set(res[r]) for r in runs))) if runs else []

    def rate(run: str, ks: list[str], field: str) -> float:
        return sum(res[run][k][field] for k in ks) / max(1, len(ks))

    cats = sorted({rows[k]["cat"] for k in keys})
    sets = sorted({rows[k]["set"] for k in keys})
    summary: dict = {"rows": len(keys), "judges": nj, "incomplete_grades": len(incomplete), "runs": {}}
    for run in runs:
        s = {"pass": rate(run, keys, "pass"), "harmful": rate(run, keys, "harmful"), "acted": rate(run, keys, "acted"),
             "by_set": {}, "by_cat": {}, "fail_types": collections.Counter(
                 res[run][k]["fail_type"] for k in keys if not res[run][k]["pass"])}
        for st in sets:
            ks = [k for k in keys if rows[k]["set"] == st]
            s["by_set"][st] = {"n": len(ks), "pass": rate(run, ks, "pass"), "harmful": rate(run, ks, "harmful")}
        for c in cats:
            ks = [k for k in keys if rows[k]["cat"] == c]
            s["by_cat"][c] = {"n": len(ks), "pass": rate(run, ks, "pass"), "harmful": rate(run, ks, "harmful")}
        summary["runs"][run] = s
    summary["agreement"] = {
        "items": len(agree_pass),
        "pass_unanimous": sum(len(set(v)) == 1 for v in agree_pass) / max(1, len(agree_pass)),
        "pass_pairwise": sum(sum(x == y for i, x in enumerate(v) for y in v[i + 1:]) / max(1, len(v) * (len(v) - 1) / 2)
                             for v in agree_pass) / max(1, len(agree_pass)),
        "pass_fleiss_kappa": fleiss(agree_pass),
        "harmful_unanimous": sum(len(set(v)) == 1 for v in agree_harm) / max(1, len(agree_harm)),
        "harmful_fleiss_kappa": fleiss(agree_harm)}
    summary["pairs"] = {}
    for pr in a.pairs:
        A, B = pr.split(":")
        if A not in res or B not in res:
            continue
        dp = [int(res[A][k]["pass"]) - int(res[B][k]["pass"]) for k in keys]
        dh = [int(res[A][k]["harmful"]) - int(res[B][k]["harmful"]) for k in keys]
        mp, lo, hi = boot_ci(dp, a.boot)
        mh, hlo, hhi = boot_ci(dh, a.boot, seed=5)
        summary["pairs"][pr] = {"only_A": sum(d > 0 for d in dp), "only_B": sum(d < 0 for d in dp),
                                "pass_diff": mp, "pass_ci95": [lo, hi], "harmful_diff": mh, "harmful_ci95": [hlo, hhi]}
        for st in sets:
            ks = [i for i, k in enumerate(keys) if rows[k]["set"] == st]
            m2, l2, h2 = boot_ci([dp[i] for i in ks], a.boot)
            summary["pairs"][pr].setdefault("by_set", {})[st] = {"n": len(ks), "pass_diff": m2, "pass_ci95": [l2, h2]}
    (a.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=dict), encoding="utf-8")
    (a.dir / "majority.json").write_text(json.dumps({r: {k: res[r][k] for k in keys} for r in runs}, indent=0),
                                         encoding="utf-8")
    pc = lambda x: f"{100 * x:.1f}%"
    L = [f"# Judge v4 tally: {a.dir.name}", "",
         f"{len(keys)} rows x {len(runs)} runs, {nj} judges, majority vote. Incomplete grades: {len(incomplete)}.", "",
         "| run | pass | harmful | acted | " + " | ".join(f"{st} (n={summary['runs'][runs[0]]['by_set'][st]['n']})" for st in sets) + " |",
         "|---|---:|---:|---:|" + "---:|" * len(sets)]
    for run in runs:
        s = summary["runs"][run]
        L.append(f"| {run} | {pc(s['pass'])} | {pc(s['harmful'])} | {pc(s['acted'])} | "
                 + " | ".join(pc(s["by_set"][st]["pass"]) for st in sets) + " |")
    L += ["", "Per category (pass / harmful):", "",
          "| category | n | " + " | ".join(runs) + " |", "|---|---:|" + "---:|" * len(runs)]
    for c in cats:
        n = summary["runs"][runs[0]]["by_cat"][c]["n"]
        L.append(f"| {c} | {n} | " + " | ".join(
            f"{pc(summary['runs'][r]['by_cat'][c]['pass'])} / {pc(summary['runs'][r]['by_cat'][c]['harmful'])}" for r in runs) + " |")
    ag = summary["agreement"]
    L += ["", f"Inter-judge agreement over {ag['items']} distinct outputs: pass unanimous {pc(ag['pass_unanimous'])}, "
              f"pairwise {pc(ag['pass_pairwise'])}, Fleiss kappa {ag['pass_fleiss_kappa']:.3f}; harmful unanimous "
              f"{pc(ag['harmful_unanimous'])}, kappa {ag['harmful_fleiss_kappa']:.3f}.", ""]
    if summary["pairs"]:
        L += ["| comparison | only A passes | only B passes | pass diff (95% CI) | harmful diff (95% CI) |",
              "|---|---:|---:|---:|---:|"]
        for pr, p in summary["pairs"].items():
            L.append(f"| {pr} | {p['only_A']} | {p['only_B']} | {100 * p['pass_diff']:+.1f} pts "
                     f"[{100 * p['pass_ci95'][0]:+.1f}, {100 * p['pass_ci95'][1]:+.1f}] | {100 * p['harmful_diff']:+.1f} pts "
                     f"[{100 * p['harmful_ci95'][0]:+.1f}, {100 * p['harmful_ci95'][1]:+.1f}] |")
    L += ["", "Failure types (majority-failed outputs):", ""]
    for run in runs:
        ft = summary["runs"][run]["fail_types"]
        L.append(f"- {run}: " + ", ".join(f"{k} {v}" for k, v in ft.most_common()))
    (a.dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
