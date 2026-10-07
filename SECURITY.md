# Security policy

## Reporting a vulnerability

Please **do not** open a public issue, discussion or pull request for a security problem.

Report it privately through GitHub: go to the
[Security tab](https://github.com/jordan-gibbs/ochre/security) and click
**Report a vulnerability** (GitHub Security Advisories). Only the maintainer can see the report.

Include what you can: affected version or commit, OS, steps to reproduce or a proof of concept,
and the impact you expect. You'll get an acknowledgement within a few days. Ochre has a single
maintainer, so please allow reasonable time for a fix before any public disclosure; we will
agree on a date together and credit you in the advisory unless you prefer otherwise.

## Supported versions

Only the latest release and `main` receive security fixes.

## Scope

In scope, for example:

- **API key handling:** keys for cloud speech / LLM providers leaking into logs, history,
  `timings.jsonl`, crash output, the config file in plain text where the OS keychain should be
  used, error messages, or network requests to the wrong host.
- **Injection:** text injection (SendInput / CGEvent / xdotool) typing into the wrong window or
  triggering keystrokes beyond the dictated text; prompt injection in dictated text that makes
  the cleanup model act on the text instead of cleaning it, in a way that has a security impact;
  command or argument injection in the llama-server sidecar launch.
- **Model and binary downloads:** missing or bypassable SHA-256 verification, path traversal
  when unpacking archives, downloads over plain HTTP, redirects to untrusted hosts, or a way to
  make Ochre run a binary it did not verify.
- **Local IPC:** the Tauri bridge or the sidecar's local HTTP port being reachable from other
  users, other machines or web pages.
- **Privacy:** audio or transcripts leaving the machine when the user chose a local engine.

Out of scope: vulnerabilities in third-party services (report them to the provider),
attacks that need an already-compromised account or machine, and issues in model weights'
outputs that have no security impact.
