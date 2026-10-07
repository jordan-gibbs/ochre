# Going public: release checklist

The exact steps to take `jordan-gibbs/ochre` and the `polonuim210/ochre-refine-*` models public
and ship v0.1.0. Run them in order. The pre-publication audit (2026-10-05) found **no secrets
in git history or the working tree** (gitleaks 8.30.1 over all 43 commits, plus regexes for
`sk-`, `hf_`, `AKIA`, `ghp_`, `dtn_`, `xox*`, private keys and bearer tokens; the only hits are
obviously fake test fixtures), and **no file over 5 MB in history** (the largest blob is a
1.7 MB training JSONL).

## 1. Secrets

- [ ] Nothing to rotate from the audit. If anything was committed since, re-run before going
      public:
      ```sh
      gitleaks git --log-opts="--all" --redact -v .
      ```
      If that finds anything: rotate the key first, then decide on a history rewrite
      (`git filter-repo`) **before** step 3. After the repo is public, assume anything in
      history is leaked.
- [ ] Note: the Daytona organization id (a UUID, not a credential) and absolute local paths under
      your home directory remain in **history** (commits before
      2026-10-05). They are redacted in the working tree. Rewriting history only for these is
      optional.
- [x] `assets/wake/transcribe.onnx` had PyTorch export stack traces (node metadata) with local
      paths. Stripped 2026-10-06; weights unchanged, parity tests pass, sha256 updated.

## 2. Private / internal documents

**Done 2026-10-06:** references to the maintainer's other projects, personal paths and internal
notes were removed from the tree, and the export stack traces (with local paths) were stripped
from `assets/wake/transcribe.onnx`. Older versions remain in git history unless the public repo
starts from a fresh commit.

### 2b. Licence questions only you can answer

- [ ] **Qwen3-TTS voice clone reference.** 44 real-v2 rows were rendered by cloning a reference
      voice (`clone-a`, `CLONE_REF`). Confirm that reference audio is yours to use (not a
      cloned real person or a voice whose terms forbid cloning). Only transcripts are published,
      not audio.
- [x] **Wake word training data.** Decided 2026-10-06: the model is CC BY-NC-SA 4.0
      (`assets/wake/LICENSE`), the code stays MIT. `assets/wake/transcribe.onnx` was trained with
      livekit-wakeword / openWakeWord data: precomputed ACAV100M features, free-sound
      background noise and MIT room impulse responses. openWakeWord ships its *own* pre-trained
      heads under CC BY-NC-SA 4.0 partly because of such training data. Decide whether you're
      comfortable releasing this head under MIT (THIRD_PARTY_NOTICES.md says MIT today) or
      whether to label it CC BY-NC-SA 4.0 / retrain on clearly licensed negatives.
- [ ] **LLM-written labels.** Scripts and targets were written / audited with Claude (Opus,
      Sonnet). Check that your plan's terms permit publishing a fine-tune trained on those
      outputs (a narrow dictation-cleanup model is unlikely to be a "competing model", but it is
      your call).
- [ ] TTS voices otherwise look fine for publishing derived text: Piper LibriTTS-R voices
      (CC-BY-4.0 data), Kokoro-82M (Apache-2.0), Qwen3-TTS (Apache-2.0), Windows SAPI
      David/Zira (Windows licence; no audio redistributed). Parakeet is CC-BY-4.0 (attribution
      is in THIRD_PARTY_NOTICES.md).

## 3. Make the GitHub repo public

```sh
gh repo edit jordan-gibbs/ochre --visibility public --accept-visibility-change-consequences
```

## 4. Protection and security settings

Already applied while private on 2026-10-05: ruleset `main` (id 24524324, active), squash-only
merges, delete branch on merge, auto-merge, Discussions, Dependabot alerts + security updates.
Re-run the script after going public to turn on what GitHub only allows on public repos
(private vulnerability reporting, secret scanning + push protection). It is idempotent:

```sh
scripts/github-protect.sh --dry-run   # see what will be sent
scripts/github-protect.sh
```

Check: `gh api repos/jordan-gibbs/ochre --jq .security_and_analysis` shows secret scanning and
push protection `enabled`, and https://github.com/jordan-gibbs/ochre/security shows
"Report a vulnerability".

- [ ] Open a throwaway PR (e.g. a typo fix) and confirm the five required checks run and pass:
      `fmt`, `check (ubuntu-latest)`, `check (windows-latest)`, `check (macos-latest)`,
      `ui tests`. If a job name changes in `.github/workflows/ci.yml`, update `CHECKS` in the
      script and re-run it.

## 5. Make the Hugging Face models public

**Done 2026-10-06** (all three repos public; anonymous download verified).

The model cards already have `license: apache-2.0`, `base_model: Qwen/Qwen3.5-*` and link to
github.com/jordan-gibbs/ochre. (`ochre-refine-2b` also still holds the old
`ochre-refine-2b-v2-Q4_K_M.gguf`; delete it first if you don't want v2 public.)

```sh
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for n in ("4b", "2b", "0.8b"):
    api.update_repo_settings(f"polonuim210/ochre-refine-{n}", private=False)
    print(n, "private =", api.model_info(f"polonuim210/ochre-refine-{n}").private)
EOF
curl -sI https://huggingface.co/polonuim210/ochre-refine-2b/resolve/main/ochre-refine-2b-v4-Q4_K_M.gguf | head -1   # expect 302, not 401
```

## 6. Default to Ochre Refine for everyone (code change, separate PR)

**Done 2026-10-06** (`auto` always picks Ochre Refine; Quill is only the download fallback).

Today `auto` only picks Ochre Refine when an HF token is configured or the file is already
downloaded, and falls back to Quill otherwise. Once the repos are public:

- `crates/ochre-refine/src/local/auto.rs`
  - `choose()`: drop the `has_token || downloaded(ochre)` condition and always return
    `model_for(t, true)`; `resolve()` then no longer needs `ochre_models::has_hf_token()`.
  - Keep `public_equivalent()` + the fallback in `ensure()` so an offline / failed download
    still falls back to the same-size Quill model.
  - Update the module docs (lines 13-15) and the tests `falls_back_to_quill_without_access`
    and the "With access to the private models" case.
- `crates/ochre-refine/src/local/install.rs`: the "PRIVATE for now" comment on the
  `OCHRE_REFINE_*_REPO` constants and the `DEFAULT_MODEL` comment.
- `crates/ochre-refine/src/local/mod.rs`: module docs (line 18-19) and the "private for now
  (needs HF_TOKEN)" text in the provider description (line ~62).
- `crates/ochre-models/src/lib.rs`: the ignored live test (`#[ignore = "network + HF token ..."]`
  near line 676) asserts an anonymous HEAD gets 401/403; flip it to expect 200.
- `README.md`: "The Ochre Refine models are private for now ..." sentence in the features list.

## 7. Tag and release v0.1.0

```sh
git switch main && git pull
git tag -a v0.1.0 -m "Ochre v0.1.0"
git push origin v0.1.0
```

The tag runs `.github/workflows/release.yml`, which builds the Windows setup `.exe`, the macOS
`.dmg` and the Linux `.deb` / AppImage and attaches them to a **draft** release. Download and
smoke-test each one, edit the notes, then publish the draft. To try the build without a tag, run
the workflow by hand (Actions → Release → Run workflow) and grab the artifacts.

## 8. Community

- [ ] Discussions are enabled (done by the script). Optionally pin a welcome post and create
      categories (Q&A, Ideas, Show and tell).
- [ ] Add repo topics and description:
      `gh repo edit jordan-gibbs/ochre --add-topic dictation,speech-to-text,rust,tauri,local-first,parakeet`
