"""Round-4 DPO pairs (Phase E): chosen = the audited target, rejected = a v3 model's own output.

    python tools/refine-data/build_dpo_v4.py --outputs tools/eval/out/eval-runs/<run>/refine-v3-2b-trainreal.jsonl
    python tools/refine-data/build_dpo_v4.py ... --sample 150     # also writes the pairs for the audit check

Input: per-row eval outputs of a v3 model on `real-v4/train-real.jsonl` (training rows only; eval
rows never become pairs). A pair is kept only when the v3 output is a *clear* failure, found
mechanically (no judge), so `rejected` is worse than `chosen` without needing a grade:

* invented: the output has a content word that is in neither `raw` nor the target (the top
  harmful failure of round 3: misrecognition guessed / content made up);
* dropped: the output loses >= 2 content words that are in both `raw` and the target;
* unfixed correction / filler: the row is tagged self_correction or the target is much shorter
  than raw, and the output keeps >= 2 content words the target removed (missed correction).

Pure punctuation/casing differences are never pairs; neither are number renderings, units
("km"), joins/splits ("co-op"/"coop", "getUser"/"get_user") or inflections ("opened"/"opens").
A 150-pair sample of the first filter version (`dpo-audit/sample-v1.jsonl`) was checked by two
auditors (`dpo-audit/A.jsonl` Opus, `B.jsonl` Sonnet): 124 "worse" by both; with the tightened
filter, 82 of the 89 surviving sample pairs are "worse" by both (92%). Audited pairs override the
filter (kept iff both said "worse"). Writes `real-v4/dpo-pairs.jsonl` rows in the
training format plus `chosen`, `rejected`, `fail` (the reasons) and `source: dpo-v4`.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "training" / "refine" / "datasets"
FUNC = set("""a an the and or but so if then than that this these those there here of to in on at by for with
from up down out over into onto about as is are was were be been being am do does did have has had i you he
she it we they me him her us them my your his its our their what which who when where why how not no yes just
can could will would shall should may might must also too very really like well oh okay ok all any some each
every one more most other such only um uh hmm yeah hi hey let lets let's i'm it's that's don't can't won't i'll we'll
you're we're they're i've gonna wanna actually basically mean know sorry wait""".split())


# Number words, units and their abbreviations: number rendering is a style choice (rule 7), never a pair.
NUM = set("""zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
million billion half quarter past point percent dollar dollars cent cents euro euros pound pounds first second
third fourth fifth sixth seventh eighth ninth tenth km kilometer kilometers kilometres kg kilogram kilograms
mi mile miles lb lbs ft feet inch inches min mins minute minutes hr hrs hour hours sec secs pm am p.m a.m""".split())


def cw(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))
            if w not in FUNC and w not in NUM and len(w) > 1 and not re.search(r"\d", w)]


def joined(s: str) -> str:
    """Letters only: makes co-op / coop, drop off / dropoff, getUser / get_user compare equal."""
    return re.sub(r"[^a-z]", "", s.lower())


def stem(w: str) -> str:
    w = w.replace("'", "")
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def classify(raw: str, clean: str, out: str, tags: list[str]) -> list[str]:
    r, c, o = set(cw(raw)), collections.Counter(cw(clean)), collections.Counter(cw(out))
    stems = {stem(w) for w in r | set(c)}
    jr, jc, jo = joined(raw), joined(clean), joined(out)
    fails = []
    # an "invented" word that is only a join/split or an inflection of a raw/target word is not invented
    inv = [w for w in o if w not in r and w not in c and stem(w) not in stems
           and joined(w) not in jr and joined(w) not in jc]
    if inv:
        fails.append("invented:" + ",".join(sorted(inv)[:4]))
    lost = [w for w in c if w in r and o[w] < c[w] and stem(w) not in {stem(x) for x in o}
            and not (joined(w) in jo and len(w) >= 3)]
    if sum(c[w] - o[w] for w in lost) >= 2:
        fails.append("dropped:" + ",".join(sorted(lost)[:4]))
    removed = [w for w in r if w not in c]
    kept_removed = [w for w in removed if w in o and joined(w) not in jc]
    if ("self_correction" in tags or len(cw(clean)) < 0.8 * len(cw(raw))) and len(kept_removed) >= 2:
        fails.append("unfixed:" + ",".join(sorted(kept_removed)[:4]))
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--train", default=str(DS / "real-v4" / "train-real.jsonl"))
    ap.add_argument("--out", default=str(DS / "real-v4" / "dpo-pairs.jsonl"))
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    train = {}
    for line in Path(a.train).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            r = json.loads(line)
            train[r["id"]] = r
    pairs, why = [], collections.Counter()
    seen = set()
    for f in a.outputs:
        for line in Path(f).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            r = train.get(o["id"])
            out = (o.get("output") or "").strip()
            if r is None or not out or o.get("em_norm") or (r["id"], out) in seen:
                continue
            fails = classify(r["raw"], r["clean"], out, r.get("tags") or [])
            if not fails:
                continue
            seen.add((r["id"], out))
            for x in fails:
                why[x.split(":")[0]] += 1
            pairs.append({**{k: r[k] for k in ("mode", "style", "dictionary", "context", "raw", "tags") if k in r},
                          "id": f"dpo4-{r['id']}", "source": "dpo-v4", "row_id": r["id"],
                          "chosen": r["clean"], "rejected": out, "rejected_by": o.get("system", ""),
                          "fail": fails})
    # Audited pairs override the filter: both auditors said "worse" -> kept, otherwise removed.
    ad = Path(a.out).parent / "dpo-audit"
    if all((ad / f).exists() for f in ("sample-v1.jsonl", "A.jsonl", "B.jsonl")):
        def verdicts(f: str) -> dict:
            return {r["id"]: r["verdict"] for r in map(json.loads, (ad / f).read_text(encoding="utf-8-sig").splitlines())}
        va, vb = verdicts("A.jsonl"), verdicts("B.jsonl")
        audited = [json.loads(x) for x in (ad / "sample-v1.jsonl").read_text(encoding="utf-8-sig").splitlines() if x]
        good = {r["id"]: r for r in audited if va.get(r["id"]) == "worse" == vb.get(r["id"])}
        aud_ids = {r["id"] for r in audited}
        n0 = len(pairs)
        pairs = [p for p in pairs if p["id"] not in aud_ids or p["id"] in good]
        have = {p["id"] for p in pairs}
        pairs += [dict(r, audited=True) for i, r in good.items() if i not in have]
        print(f"audit override: {n0} -> {len(pairs)} pairs ({len(good)} of {len(audited)} audited pairs 'worse' by both)")
    Path(a.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    print(f"{len(pairs)} pairs -> {a.out}; reasons {dict(why)}")
    if a.sample:
        pick = random.Random(4).sample(pairs, min(a.sample, len(pairs)))
        sp = Path(a.out).with_name("dpo-audit-sample.jsonl")
        sp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pick), encoding="utf-8")
        print(f"{len(pick)} sampled -> {sp}")


if __name__ == "__main__":
    main()
