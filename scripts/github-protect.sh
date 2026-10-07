#!/usr/bin/env bash
# Protect `main` and set merge / security settings for the public repo. Idempotent: safe to
# re-run; an existing ruleset with the same name is updated in place.
#
#   scripts/github-protect.sh            # apply
#   scripts/github-protect.sh --dry-run  # print what would be sent, change nothing
#
# Needs: gh (authenticated as a repo admin), jq is NOT required.
# Run it right after the repo is made public: GitHub refuses rulesets / branch protection on
# private repos of free personal accounts ("Upgrade to GitHub Pro"), and secret scanning +
# push protection are free only on public repos.
#
# What it sets:
#   * Ruleset "main" on the default branch:
#       - pull request required (no approval: a solo maintainer can't approve their own PRs),
#         all conversations resolved, squash merge only
#       - required status checks (not strict: re-running CI on every open PR after each merge
#         cost more than it caught) = the CI job names in
#         .github/workflows/ci.yml, pinned to the GitHub Actions app
#       - no force pushes, no deletion, linear history
#       - bypass: repository admins (the maintainer), "always", so the owner can still push /
#         merge their own work (they cannot approve their own PRs). Equivalent of enforce_admins=false.
#   * Merge settings: squash only (title = PR title, body = PR body), delete branch on merge,
#     auto-merge allowed, "update branch" button.
#   * Discussions on; private vulnerability reporting, Dependabot alerts + security updates,
#     secret scanning + push protection on.
set -euo pipefail

REPO="${REPO:-jordan-gibbs/ochre}"
RULESET_NAME="main"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# Job names from .github/workflows/ci.yml (the `name:` GitHub reports as the check context).
CHECKS=("fmt" "check (ubuntu-latest)" "check (windows-latest)" "check (macos-latest)" "ui tests")
ACTIONS_APP_ID=15368   # the "GitHub Actions" app: only checks it reports count

say() { printf '==> %s\n' "$*"; }

# api METHOD PATH [json-body]
api() {
  local method="$1" path="$2" body="${3:-}"
  if (( DRY )); then
    printf -- '--- gh api -X %s %s\n' "$method" "$path"
    [[ -n "$body" ]] && printf '%s\n' "$body"
    return 0
  fi
  if [[ -n "$body" ]]; then
    gh api -X "$method" "$path" -H "Accept: application/vnd.github+json" --input - <<<"$body" >/dev/null
  else
    gh api -X "$method" "$path" -H "Accept: application/vnd.github+json" >/dev/null
  fi
}

checks_json=""
for c in "${CHECKS[@]}"; do
  checks_json+="${checks_json:+,}{\"context\":\"$c\",\"integration_id\":$ACTIONS_APP_ID}"
done

ruleset=$(cat <<JSON
{
  "name": "$RULESET_NAME",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "bypass_actors": [
    { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always" }
  ],
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_linear_history" },
    { "type": "pull_request", "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "allowed_merge_methods": ["squash"]
    } },
    { "type": "required_status_checks", "parameters": {
        "strict_required_status_checks_policy": false,
        "do_not_enforce_on_create": false,
        "required_status_checks": [ $checks_json ]
    } }
  ]
}
JSON
)

say "repo $REPO$( (( DRY )) && echo ' (dry run)')"
if (( ! DRY )); then
  vis=$(gh api "repos/$REPO" --jq .visibility)
  [[ "$vis" == "public" ]] || say "WARNING: repo is $vis; rulesets may be refused (needs public or GitHub Pro)"
fi

say "merge settings + discussions"
api PATCH "repos/$REPO" '{
  "allow_squash_merge": true,
  "allow_merge_commit": false,
  "allow_rebase_merge": false,
  "squash_merge_commit_title": "PR_TITLE",
  "squash_merge_commit_message": "PR_BODY",
  "delete_branch_on_merge": true,
  "allow_auto_merge": true,
  "allow_update_branch": true,
  "has_discussions": true
}'

say "ruleset \"$RULESET_NAME\" on the default branch"
existing=""
if (( ! DRY )); then
  existing=$(gh api "repos/$REPO/rulesets" --jq ".[] | select(.name == \"$RULESET_NAME\") | .id" || true)
fi
if [[ -n "$existing" ]]; then
  api PUT "repos/$REPO/rulesets/$existing" "$ruleset"
  say "updated ruleset $existing"
else
  api POST "repos/$REPO/rulesets" "$ruleset"
  say "created ruleset"
fi

say "security: private vulnerability reporting, Dependabot alerts + fixes"
# These need a public repo (or GitHub Advanced Security); warn and carry on if refused.
soft() { api "$@" || say "  refused: $1 $2 (expected while the repo is private; re-run after going public)"; }
soft PUT "repos/$REPO/private-vulnerability-reporting"
soft PUT "repos/$REPO/vulnerability-alerts"
soft PUT "repos/$REPO/automated-security-fixes"

say "security: secret scanning + push protection"
soft PATCH "repos/$REPO" '{
  "security_and_analysis": {
    "secret_scanning": { "status": "enabled" },
    "secret_scanning_push_protection": { "status": "enabled" }
  }
}'

if (( ! DRY )); then
  say "result"
  gh api "repos/$REPO" --jq '{visibility, allow_squash_merge, allow_merge_commit, allow_rebase_merge, delete_branch_on_merge, allow_auto_merge, has_discussions, security: .security_and_analysis}'
  gh api "repos/$REPO/rulesets" --jq '.[] | {id, name, enforcement}'
fi
say "done"
