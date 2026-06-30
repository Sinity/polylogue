#!/usr/bin/env bash
set -euo pipefail
repo=${1:?usage: gh_pr_safety.sh OWNER/REPO PR_NUMBER BRANCH_NAME}
pr=${2:?usage: gh_pr_safety.sh OWNER/REPO PR_NUMBER BRANCH_NAME}
branch=${3:?usage: gh_pr_safety.sh OWNER/REPO PR_NUMBER BRANCH_NAME}
remote_sha=$(git ls-remote "https://github.com/${repo}.git" "refs/heads/${branch}" | awk '{print $1}')
pr_sha=$(gh pr view "$pr" -R "$repo" --json headRefOid --jq .headRefOid)
state=$(gh pr view "$pr" -R "$repo" --json mergeStateStatus,mergeable --jq '{mergeable,mergeStateStatus}')
echo "remote branch: ${remote_sha}"
echo "PR headRefOid: ${pr_sha}"
echo "PR state: ${state}"
if [[ -z "$remote_sha" ]]; then
  echo "ERROR: remote branch not found" >&2
  exit 2
fi
if [[ "$remote_sha" != "$pr_sha" ]]; then
  echo "ERROR: PR object is decoupled from branch ref; close/recreate PR before trusting mergeability" >&2
  exit 3
fi
