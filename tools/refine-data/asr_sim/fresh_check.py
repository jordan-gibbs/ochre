"""Sim-vs-real check on the fresh pairs (W12-W51): v3-4b word error vs target, real raw vs simulated raw,
keeping only simulated samples that distill_v4.patch_target accepts (target recoverable). Run from the repo root
after `owf_daytona.py eval --models v3-4b=... --eval freshval=datasets/asr-sim/fresh-pairs.jsonl`;
edit the output path below to that run."""
import json,sys,statistics as st
sys.path.insert(0,"tools/refine-data")
import distill_v4 as dv
mp=json.load(open("tools/refine-data/asr_sim/fresh_pairs_map.json",encoding="utf-8"))
rows={r["id"]:r for r in map(json.loads,open("tools/eval/out/eval-runs/eval-1005-155709-f02f/refine-v3-4b-freshval.jsonl",encoding="utf-8"))}
pairs={r["id"]:r for r in map(json.loads,open("training/refine/datasets/asr-sim/fresh-pairs.jsonl",encoding="utf-8"))}
def wer(a,b):
    a,b=dv.words(a),dv.words(b)
    ops=dv.align(a,b); return sum(o[0]!="eq" for o in ops)/max(1,len(a))
acc=[];R=[];Sm=[];emr=[];ems=[]
for n,m in mp.items():
    s=rows[f"simval-sim-{n}"]; r=rows[f"simval-real-{n}"]; p=pairs[f"simval-sim-{n}"]
    tgt,info=dv.patch_target(m["tts_text"],p["raw"],p["clean"],p.get("dictionary") or [])
    if tgt is None: continue
    acc.append(n); R.append(wer(r["clean"],r["output"])); Sm.append(wer(tgt,s["output"]))
    emr.append(dv.words(r["clean"])==dv.words(r["output"])); ems.append(dv.words(tgt)==dv.words(s["output"]))
print("recoverable sim pairs",len(acc),"of",len(mp))
print("real: wer %.4f  word-EM %.3f"%(st.mean(R),st.mean(emr)))
print("sim (patched target): wer %.4f  word-EM %.3f"%(st.mean(Sm),st.mean(ems)))
