"""LoRA SFT of Qwen3.5 for dictation cleanup (training/refine/README.md).

Every example is rendered with render.py, byte-identical to the production local prompt (checked
against the Rust renderer before training), so the model trains on exactly what llama-server sends
it: raw ChatML ending in the pre-seeded empty think block. Loss is on the completion
(``clean + <|im_end|>``) only. Logits are computed only at completion positions, which saves the
~250k-vocab logits for the ~400-token system prompt.

    .venv-train-refine/Scripts/python train.py configs/qwen35-2b-lora.yaml
    .venv-train-refine/Scripts/python train.py configs/qwen35-2b-lora.yaml --set epochs=1 --set run_name=dry
    .venv-train-refine/Scripts/python train.py configs/qwen35-2b-lora.yaml --resume       # last checkpoint

Optional DPO stage (round 4, ``dpo_files`` set): after SFT (or starting from ``init_adapter`` with
``epochs: 0``), preference pairs ``{raw, context, style, dictionary, mode, chosen, rejected}`` train the
same LoRA with the DPO loss (beta ``dpo_beta``) plus ``dpo_sft_alpha`` x the chosen NLL per token (keeps
the policy anchored); the reference is the policy before the DPO stage (log-probs precomputed).

Outputs in ``output/<run_name>/``: ``checkpoint-*`` (per epoch), ``adapter/`` (final LoRA),
``config.resolved.yaml``, ``train_summary.json`` (losses, timings, throughput, peak memory).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Sequential weight loading: the threaded loader spikes host commit, which crashed on this box
# while the TTS/ASR data jobs were running.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,  # noqa: E402
                          TrainingArguments, enable_full_determinism, set_seed)

import render  # noqa: E402


# ---------------------------------------------------------------- config


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    for o in overrides:
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["output_dir"] = str(cfg["output_dir"]).format(**cfg)
    return cfg


def rel(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else HERE / q


def expand(patterns: list[str]) -> list[Path]:
    out = []
    for pat in patterns or []:
        hits = sorted(glob.glob(str(rel(pat))))
        if not hits:
            raise SystemExit(f"no files match {pat}")
        out += [Path(h) for h in hits]
    return out


# ---------------------------------------------------------------- data


def build_datasets(cfg: dict, tok) -> tuple[Dataset, Dataset | None, dict]:
    train_rows, skipped = render.load_rows(expand(cfg["train_files"]))
    eval_rows, eval_skipped = render.load_rows(expand(cfg.get("eval_files") or []))
    skipped += eval_skipped
    if not eval_rows and cfg.get("eval_split"):
        rng = random.Random(cfg["seed"])
        idx = list(range(len(train_rows)))
        rng.shuffle(idx)
        n_eval = max(1, round(len(idx) * float(cfg["eval_split"])))
        held = set(idx[:n_eval])
        eval_rows = [train_rows[i] for i in sorted(held)]
        train_rows = [r for i, r in enumerate(train_rows) if i not in held]
    for f in expand(cfg.get("forbid_files") or []):
        render.check_leakage(train_rows + eval_rows, f)
    if cfg.get("check_rust", True):
        render.check_rust(train_rows + eval_rows)
    render.check_chat_template(train_rows + eval_rows, tok)

    def encode(rows: list[dict], name: str) -> tuple[Dataset, dict]:
        recs, too_long = [], []
        for r in rows:
            t = render.tokenize(render.render(r), tok)
            if len(t["input_ids"]) > cfg["max_len"]:
                too_long.append(r["id"])
                continue
            recs.append({"input_ids": t["input_ids"], "labels": t["labels"], "length": len(t["input_ids"]),
                         "n_completion": t["n_completion"]})
        lens = [x["length"] for x in recs]
        stats = {"n": len(recs), "skipped_too_long": too_long, "tokens": sum(lens),
                 "completion_tokens": sum(x["n_completion"] for x in recs),
                 "len_mean": round(sum(lens) / max(1, len(lens)), 1), "len_max": max(lens, default=0)}
        print(f"data[{name}]: {stats['n']} examples, {stats['tokens']} tokens "
              f"({stats['completion_tokens']} supervised), mean len {stats['len_mean']}, max {stats['len_max']}"
              + (f", {len(too_long)} skipped > max_len {cfg['max_len']}" if too_long else ""))
        return Dataset.from_list(recs), stats

    train_ds, tstats = encode(train_rows, "train")
    eval_ds, estats = encode(eval_rows, "eval") if eval_rows else (None, {})
    return train_ds, eval_ds, {"train": tstats, "eval": estats, "skipped_rows": skipped}


class Collator:
    """Right padding (safe for the recurrent layers: padding comes after every real token)."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), n), -100, dtype=torch.long)
        att = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            lab[i, :k] = torch.tensor(f["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ---------------------------------------------------------------- model


def attach_lora(model, cfg: dict):
    targets = list(cfg["target_modules"])
    # Only the text decoder layers (the MTP head and vision tower are not loaded by *ForCausalLM).
    pattern = r"^model\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\.(?:" + "|".join(map(re.escape, targets)) + r")$"
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=pattern, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    # Verify coverage of the hybrid architecture: every layer's mixer and MLP got adapters.
    layer_types = model.get_base_model().config.layer_types
    hit: dict[int, set[str]] = {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.(self_attn|linear_attn|mlp)\.(\w+)$", name)
        if m and hasattr(mod, "lora_A"):
            hit.setdefault(int(m.group(1)), set()).add(f"{m.group(2)}.{m.group(3)}")
    for i, lt in enumerate(layer_types):
        mixer = "self_attn" if lt == "full_attention" else "linear_attn"
        got = hit.get(i, set())
        if not any(x.startswith(mixer) for x in got) or not any(x.startswith("mlp.") for x in got):
            raise SystemExit(f"layer {i} ({lt}) has no LoRA on its {mixer} or mlp: {sorted(got)}")
    n_lin = sum(1 for t in layer_types if t == "linear_attention")
    ex_lin = sorted(hit[layer_types.index("linear_attention")])
    ex_att = sorted(hit[layer_types.index("full_attention")])
    print(f"lora: {sum(len(v) for v in hit.values())} modules over {len(hit)} layers "
          f"({n_lin} linear_attention, {len(layer_types) - n_lin} full_attention)\n"
          f"  linear_attention layer: {ex_lin}\n  full_attention layer:   {ex_att}")
    model.print_trainable_parameters()
    return model


class SFTTrainer(Trainer):
    """Loss = CE over completion tokens only, logits computed only at those positions."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model_accepts_loss_kwargs = True   # we normalise by num_items_in_batch ourselves

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        labels = inputs["labels"]
        hidden = base.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            use_cache=False).last_hidden_state
        tgt = labels[:, 1:]
        sel = tgt != -100
        h = hidden[:, :-1][sel]
        logits = base.lm_head(h).float()
        loss_sum = torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="sum")
        if num_items_in_batch is not None and self.model.training:
            denom = num_items_in_batch.to(loss_sum.device) if torch.is_tensor(num_items_in_batch) else num_items_in_batch
        else:
            denom = sel.sum().clamp(min=1)
        loss = loss_sum / denom
        return (loss, {}) if return_outputs else loss


class Guard(TrainerCallback):
    """Abort on NaN/inf loss; record per-epoch wall clock and eval loss."""

    def __init__(self) -> None:
        self.epoch_t0 = None
        self.epochs: list[dict] = []
        self.log: list[dict] = []

    def on_epoch_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.epoch_t0 = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.epochs.append({"epoch": round(state.epoch or 0, 3), "train_s": round(time.perf_counter() - self.epoch_t0, 1),
                            "step": state.global_step})

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        self.log.append({"step": state.global_step, **logs})
        for k in ("loss", "eval_loss", "grad_norm"):
            v = logs.get(k)
            if v is not None and not math.isfinite(float(v)):
                raise RuntimeError(f"non-finite {k}={v} at step {state.global_step}; aborting")


# ---------------------------------------------------------------- DPO (optional second stage)


def load_pairs(cfg: dict, tok) -> list[dict]:
    pairs = []
    for f in expand(cfg.get("dpo_files") or []):
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("chosen") or not r.get("rejected") or r["chosen"].strip() == r["rejected"].strip():
                continue
            enc = {}
            for side in ("chosen", "rejected"):
                row = {**r, "clean": r[side], "id": r.get("id", "pair")}
                tk = render.tokenize(render.render(row), tok)
                enc[side] = (tk["input_ids"], tk["labels"])
            if max(len(enc["chosen"][0]), len(enc["rejected"][0])) > cfg["max_len"]:
                continue
            pairs.append({"id": r.get("id"), "chosen": enc["chosen"], "rejected": enc["rejected"]})
    return pairs


def completion_logps(model, seqs: list, pad_id: int):
    """Sum of completion-token log-probs per sequence, and the completion token counts."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    n = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long, device="cuda")
    lab = torch.full((len(seqs), n), -100, dtype=torch.long, device="cuda")
    att = torch.zeros((len(seqs), n), dtype=torch.long, device="cuda")
    for i, (x, y) in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    hidden = base.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100
    row_of = torch.arange(len(seqs), device="cuda").unsqueeze(1).expand_as(tgt)[sel]
    logits = base.lm_head(hidden[:, :-1][sel]).float()
    lp = torch.log_softmax(logits, -1).gather(1, tgt[sel].unsqueeze(1)).squeeze(1)
    out = torch.zeros(len(seqs), device="cuda").index_add(0, row_of, lp)
    cnt = torch.zeros(len(seqs), device="cuda").index_add(0, row_of, torch.ones_like(lp))
    return out, cnt


def run_dpo(model, tok, cfg: dict) -> dict:
    pairs = load_pairs(cfg, tok)
    if not pairs:
        raise SystemExit("dpo_files gave no usable pairs")
    beta, alpha = float(cfg.get("dpo_beta", 0.1)), float(cfg.get("dpo_sft_alpha", 0.2))
    mb, accum = int(cfg.get("dpo_micro_batch", 4)), int(cfg.get("dpo_grad_accum", 4))
    epochs = int(cfg.get("dpo_epochs", 1))
    lr = float(cfg.get("dpo_lr", 5e-6))
    pad = tok.pad_token_id
    rng = random.Random(cfg["seed"] + 7)
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():   # reference = the policy before this stage
        for i in range(0, len(pairs), mb):
            chunk = pairs[i:i + mb]
            rc, _ = completion_logps(model, [p["chosen"] for p in chunk], pad)
            rr, _ = completion_logps(model, [p["rejected"] for p in chunk], pad)
            for p, a_, b_ in zip(chunk, rc.tolist(), rr.tolist()):
                p["ref"] = (a_, b_)
    t_ref = time.perf_counter() - t0
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_total = math.ceil(len(pairs) / (mb * accum)) * epochs
    warm = max(1, int(0.1 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(1, steps_total)))
    model.train()
    log, step = [], 0
    for _ep in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for k in range(0, len(order), mb * accum):
            group = [pairs[j] for j in order[k:k + mb * accum]]
            stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0}
            for i in range(0, len(group), mb):
                chunk = group[i:i + mb]
                pc, nc = completion_logps(model, [p["chosen"] for p in chunk], pad)
                pr, _ = completion_logps(model, [p["rejected"] for p in chunk], pad)
                ref = torch.tensor([p["ref"] for p in chunk], device="cuda")
                margin = beta * ((pc - ref[:, 0]) - (pr - ref[:, 1]))
                loss = -torch.nn.functional.logsigmoid(margin).mean() + alpha * (-(pc / nc.clamp(min=1))).mean()
                (loss * len(chunk) / len(group)).backward()
                stats["loss"] += loss.item() * len(chunk) / len(group)
                stats["acc"] += (margin > 0).float().sum().item() / len(group)
                stats["margin"] += margin.sum().item() / len(group)
            torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if not math.isfinite(stats["loss"]):
                raise RuntimeError(f"non-finite DPO loss at step {step}")
            if step % 5 == 0 or step == 1:
                print(f"dpo step {step}/{steps_total} loss {stats['loss']:.4f} acc {stats['acc']:.2f} "
                      f"margin {stats['margin']:.3f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            log.append({"step": step, **{k2: round(v, 5) for k2, v in stats.items()}})
    model.eval()
    return {"pairs": len(pairs), "steps": step, "beta": beta, "sft_alpha": alpha, "epochs": epochs, "lr": lr,
            "ref_s": round(t_ref, 1), "train_s": round(time.perf_counter() - t0 - t_ref, 1), "log": log}


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", action="append", default=[], help="override a config key, e.g. --set epochs=3")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N optimizer steps")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = rel(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    t_all = time.perf_counter()

    if cfg.get("full_determinism"):
        enable_full_determinism(cfg["seed"])
    else:
        set_seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available (needs a cu128+ torch for the RTX 5070 Ti)")
    print(f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}; "
          f"free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB")

    base = rel(cfg["base_model"])
    tok = AutoTokenizer.from_pretrained(base)
    t = time.perf_counter()
    train_ds, eval_ds, dstats = build_datasets(cfg, tok)
    t_data = time.perf_counter() - t

    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if cfg.get("init_adapter"):   # continue from an existing LoRA (e.g. a shipped v3 adapter)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(rel(cfg["init_adapter"])), is_trainable=True)
        model.print_trainable_parameters()
    else:
        model = attach_lora(model, cfg)
    t_load = time.perf_counter() - t

    args = TrainingArguments(
        output_dir=str(out), run_name=cfg["run_name"], seed=cfg["seed"], data_seed=cfg["seed"],
        num_train_epochs=cfg["epochs"], max_steps=a.max_steps,
        per_device_train_batch_size=cfg["micro_batch_size"], per_device_eval_batch_size=cfg["micro_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler"], warmup_steps=float(cfg["warmup_ratio"]),
        weight_decay=cfg["weight_decay"], max_grad_norm=cfg["max_grad_norm"],
        bf16=True, optim="adamw_torch_fused",
        train_sampling_strategy="group_by_length" if cfg.get("group_by_length") else "random",
        length_column_name="length",
        eval_strategy="epoch" if eval_ds is not None else "no", save_strategy="epoch",
        save_total_limit=cfg["save_total_limit"], logging_steps=cfg["logging_steps"], logging_first_step=True,
        prediction_loss_only=True, remove_unused_columns=False, dataloader_num_workers=0,
        report_to="none", disable_tqdm=False,
    )
    guard = Guard()
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         data_collator=Collator(tok.pad_token_id), callbacks=[guard])
    torch.cuda.reset_peak_memory_stats()
    eval0 = trainer.evaluate() if eval_ds is not None else {}
    if eval0:
        print(f"eval loss before training: {eval0['eval_loss']:.4f}")
    t = time.perf_counter()
    if float(cfg["epochs"]) > 0:
        result = trainer.train(resume_from_checkpoint=True if a.resume else None)
    else:   # DPO-only run from init_adapter
        from types import SimpleNamespace
        result = SimpleNamespace(training_loss=None, global_step=0)
    t_train = time.perf_counter() - t
    dpo_summary = None
    if cfg.get("dpo_files"):
        dpo_summary = run_dpo(trainer.model, tok, cfg)
        if eval_ds is not None:
            print(f"eval loss after DPO: {trainer.evaluate().get('eval_loss')}")
    trainer.model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    losses = [x for x in guard.log if "loss" in x]
    evals = [x for x in guard.log if "eval_loss" in x]
    summary = {
        "run_name": cfg["run_name"], "base_repo": cfg.get("base_repo"), "base_revision": cfg.get("base_revision"),
        "data": dstats, "eval_loss_before": eval0.get("eval_loss"),
        "eval_loss_per_epoch": [{"epoch": e.get("epoch"), "eval_loss": e["eval_loss"]} for e in evals],
        "train_loss_first": losses[0]["loss"] if losses else None, "train_loss_last": losses[-1]["loss"] if losses else None,
        "train_loss_mean": result.training_loss, "global_steps": result.global_step,
        "epochs": guard.epochs,
        "seconds": {"data_render_tokenize_checks": round(t_data, 1), "model_load": round(t_load, 1),
                    "train_incl_eval": round(t_train, 1), "total": round(time.perf_counter() - t_all, 1)},
        "throughput": {"train_examples_per_s": round(dstats["train"]["n"] * (result.global_step and
                       trainer.state.epoch or 0) / max(1e-9, sum(e["train_s"] for e in guard.epochs) or t_train or 1e-9), 2),
                       "train_tokens_per_s": round(dstats["train"]["tokens"] * (trainer.state.epoch or 0)
                                                   / max(1e-9, sum(e["train_s"] for e in guard.epochs) or t_train))},
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "dpo": dpo_summary,
        "log": guard.log,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("log", "data")}, indent=1))
    print(f"adapter -> {out / 'adapter'}")


if __name__ == "__main__":
    main()
