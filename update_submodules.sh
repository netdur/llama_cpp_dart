#!/usr/bin/env bash
set -euo pipefail

# ensure we start from repo root
cd "$(git rev-parse --show-toplevel)"

# sync and init submodules
git submodule sync --recursive
git submodule update --init --recursive

# path of your submodule
path="src/llama.cpp"

# record before hash
before=$(git ls-tree HEAD "$path" | awk '{print $3}' | cut -c1-12 || true)

# detect default branch or fall back to master
branch=$(git -C "$path" remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')
[ -z "$branch" ] && branch=master

# ensure the branch is tracked
git submodule set-branch --branch "$branch" "$path" >/dev/null

# update to latest commit
git submodule update --remote "$path"

# record after hash
after=$(git -C "$path" rev-parse --short HEAD)

# show what happened
echo "updated $path from $before to $after"

# stage and commit
git add "$path" .gitmodules 2>/dev/null || true
if [ "$before" != "$after" ]; then
  git commit -m "update submodule $path from ${before:-none} to $after"
  echo "committed update"
else
  echo "no changes to commit"
fi
