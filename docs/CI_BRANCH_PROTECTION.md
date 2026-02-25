# CI Trust Hardening and Branch Protection

## Required checks for `main`
Set these required status checks:
- `stable-regression`
- `full-ctest`
- `lint`

## One-shot setup (maintainer)

```bash
REPO=hulohot/tiny-npu
BRANCH=main

gh api \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  repos/$REPO/branches/$BRANCH/protection \
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
```

## Verify
```bash
gh api repos/$REPO/branches/$BRANCH/protection
```
