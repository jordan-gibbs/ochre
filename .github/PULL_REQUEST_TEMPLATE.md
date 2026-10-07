## What and why

<!-- One or two sentences. Link the issue: "Fixes #123". -->

## How it was tested

<!-- OS(es) tested on, commands run, and before/after numbers for anything latency-related. -->

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --locked -- -D warnings`
- [ ] `cargo test --workspace --locked`
- [ ] `node --test app/test/ui.test.mjs` (if `app/ui` changed)
- [ ] Ran the relevant `#[ignore]`d tests (network / model) if this touches those paths

## Checklist

- [ ] One focused change; refactors are in a separate PR
- [ ] Docs / SPEC updated if behaviour changed
- [ ] No keys, personal data or large binaries committed
- [ ] New models, voices, fonts or datasets are listed in `THIRD_PARTY_NOTICES.md`

By submitting this PR I agree that my contribution is licensed under the MIT License.
