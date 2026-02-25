#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-hulohot/tiny-npu}"
BRANCH="${2:-main}"

echo "Configuring branch protection for ${REPO}:${BRANCH}"

gh api \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/branches/${BRANCH}/protection" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["stable-regression", "full-ctest", "lint"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON

echo "Done."
