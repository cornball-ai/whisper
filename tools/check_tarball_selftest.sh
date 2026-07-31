#!/usr/bin/env bash
# Negative cases for check_tarball.sh.
#
# A validator that fails OPEN is worse than no validator: it converts
# "nobody checked" into "someone checked and it was fine". Verifying only
# the clean path proves nothing about that. Each case below is a bug that
# was actually present and would have passed a contaminated tarball:
#
#   .scratch_test    the original incident (an untracked scratch directory)
#   ..scratch_test   `.[!.]*` does not match names starting with two dots
#   R.*              `grep -qw` treated the name as a regex, so this matched
#                    the allowlist entry `R` and was waved through
#   dangling symlink `[ -e ]` is false for one, so it was skipped entirely
#
# Each case is planted in a disposable copy of the package built from
# `git archive`, never in the real working tree.
#
# Usage: tools/check_tarball_selftest.sh

set -uo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validator="$repo/tools/check_tarball.sh"

# The package copy lives in $work/pkg and the log in $work/out.log -- never
# inside the copy. Writing the log into the package root contaminates the
# fixture, and every negative case then "passes" because of the log rather
# than the planted entry. (Observed while writing this.)
fresh_copy() {
  local work=$1
  mkdir -p "$work/pkg"
  git -C "$repo" archive HEAD | tar x -C "$work/pkg"
}

fail=0

# Assert the validator rejects $1 AND that its output names $1 -- rejecting
# for an unrelated reason would hide a fail-open bug.
#
# Two acceptable mechanisms. Usually the allowlist check flags the entry.
# But a dangling symlink makes `R CMD build` itself fail ("cp: cannot
# stat"), so the tarball is never produced; that is still fail-closed and
# still names the file, so it counts. What must never happen is exit 0.
expect_reject() {
  local name=$1 work=$2 reason
  if bash "$validator" "$work/pkg" >"$work/out.log" 2>&1; then
    echo "::error::FAIL-OPEN: validator passed a tree containing '${name}'"
    fail=1
    return
  fi
  if grep -Fq "unexpected top-level entry: ${name}" "$work/out.log"; then
    reason=$(grep -F "unexpected top-level entry: ${name}" "$work/out.log" |
             head -1 | sed 's/.*(\(.*\))/(\1)/')
  elif grep -Fq "R CMD build failed" "$work/out.log" &&
       grep -Fq "$name" "$work/out.log"; then
    reason="(R CMD build refused it)"
  else
    echo "::error::rejected '${name}' but nothing in the output names it:"
    sed 's/^/    /' "$work/out.log" | head -6
    fail=1
    return
  fi
  printf 'ok  rejected  %-16s %s\n' "$name" "$reason"
}

plant_dir() {
  local name=$1 work
  work=$(mktemp -d)
  fresh_copy "$work"
  mkdir -p "$work/pkg/$name"
  echo contamination >"$work/pkg/$name/payload"
  expect_reject "$name" "$work"
  rm -rf "$work"
}

plant_dangling_symlink() {
  local work
  work=$(mktemp -d)
  fresh_copy "$work"
  ln -s /nonexistent/target "$work/pkg/.dangling_test"
  expect_reject ".dangling_test" "$work"
  rm -rf "$work"
}

echo "== check_tarball.sh negative cases =="
plant_dir ".scratch_test"
plant_dir "..scratch_test"
plant_dir "R.*"
plant_dangling_symlink

# And the positive case, so a validator that rejects everything is caught too.
work=$(mktemp -d)
fresh_copy "$work"
if bash "$validator" "$work/pkg" >"$work/out.log" 2>&1; then
  echo "ok  accepted  clean tree"
else
  echo "::error::validator rejected a clean tree"
  cat "$work/out.log"
  fail=1
fi
rm -rf "$work"

if [ "$fail" -eq 0 ]; then
  echo "self-test: OK"
else
  echo "self-test: FAILED"
fi
exit "$fail"
