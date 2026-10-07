# Contributing to Ochre

Thanks for helping. Ochre is maintained by one person ([@jordan-gibbs](https://github.com/jordan-gibbs)),
who reviews every pull request. Small, focused PRs get reviewed and merged fastest.

## Before you start

- **Bugs:** open an issue with the bug template (OS, Ochre version, engine, and a
  `timings.jsonl` snippet if it is about speed).
- **Features and larger changes:** open an issue or a Discussion first, so we can agree on the
  shape before you write the code. A PR that arrives without that conversation may be declined
  even if it is good work, simply because it doesn't fit the plan.
- **Security problems:** do not open an issue. See [SECURITY.md](SECURITY.md).

## Build and test

Prerequisites: Rust 1.88+ (CI pins 1.95.0), Node 20+ for the UI tests, and on Linux the packages that
`scripts/install.sh` installs (WebKitGTK 4.1, ayatana-appindicator, ALSA, xdo, xi).

Run what CI runs before you push:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --locked -- -D warnings
cargo test --workspace --locked
node --test app/test/ui.test.mjs
```

Tests that need the network, an API key or a downloaded model are `#[ignore]`d. Run them with
`cargo test -p <crate> -- --ignored` when your change touches that path, and say so in the PR.
Live tests read keys from a dotenv file (`OCHRE_LIVE_ENV`, default `./.env`, gitignored); never
commit keys.

Handy entry points:

```sh
cargo run -p ochre-app                       # the app (add -- --demo to drive every UI state)
cargo run -p ochre-stt --example bench --release -- clip.wav
cargo run -p ochre-refine --example bench --release
```

## Pull request flow

1. Fork the repo and create a branch from `main` (`fix/hud-flicker`, `feat/linux-wayland-inject`).
2. Make one change per PR. Split refactors from behaviour changes.
3. Add or update tests for what you changed. Latency-sensitive changes should include before /
   after numbers from the bench examples (see `docs/go-checklist.md`).
4. Make sure the four commands above pass, then open a PR against `main` and fill in the template.
5. CI must be green and the maintainer must approve. PRs are squash-merged, so the PR title
   becomes the commit message: write it in the imperative ("Fix HUD flicker on resume").
   CI runs the Windows and macOS jobs only when a PR touches code for that OS, dependencies, the
   Tauri app or CI itself; the maintainer can add the `full-ci` label to run everything.

Please be patient: with a single reviewer, a review can take a few days. A polite ping after a
week is fine.

## What is in scope

- Bug fixes, especially for platforms the maintainer can't test daily (Linux distros, Wayland,
  Intel Macs, non-US keyboard layouts).
- Latency and accuracy improvements backed by measurements.
- New speech or cleanup engines that fit the privacy model: local first, cloud only with the
  user's own key, nothing sent anywhere by default.
- Accessibility, translations of UI strings, documentation.

## What is out of scope

- Telemetry, analytics, accounts or anything that phones home.
- Features that require a hosted Ochre service.
- Bundling model weights or binaries in the repository (models are downloaded and SHA-256
  verified at runtime, see `crates/ochre-models`).
- Large stylistic rewrites or dependency swaps without a prior issue.

## Model training contributions

The cleanup model pipeline lives in `training/refine` and `tools/refine-data`, the wake word in
`training/wakeword`, and the eval harness in `tools/eval`.

- Training data must be synthetic or clearly licensed for redistribution and model training.
  **Never** submit real user dictations, recordings, or anything containing personal data.
- Note the licence of any new TTS voice, dataset or base model in the PR, and update
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) if it ships with the app.
- Results must come from the blind-judge eval (`tools/eval`), with the command you ran.
- Don't commit large artefacts: datasets over a few MB, GGUF / ONNX weights and audio stay out
  of git (see `.gitignore`); link to them instead.

## Licensing of contributions

There is no CLA and no DCO sign-off. By submitting a contribution you agree that it is licensed
under the project's [MIT License](LICENSE), and that you have the right to submit it.

## Conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
