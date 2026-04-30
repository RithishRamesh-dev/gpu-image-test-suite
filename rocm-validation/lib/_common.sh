#!/usr/bin/env bash
# _common.sh — shared safe helpers sourced by all modules
# Source this ONCE from run_tests.sh before any module

# ---------------------------------------------------------------------------
# _int VAR [default]
#   Strip whitespace/newlines, return a clean integer.
#   If result is non-numeric, return default (0).
# ---------------------------------------------------------------------------
_int() {
  local v="${1:-0}"
  # collapse whitespace and newlines
  v=$(echo "$v" | tr -d '[:space:]')
  # if it looks like a plain integer, use it; else 0
  [[ "$v" =~ ^-?[0-9]+$ ]] || v="${2:-0}"
  echo "$v"
}

# ---------------------------------------------------------------------------
# _sum_lines
#   Sum all whitespace-separated integers from stdin (one per line or mixed).
#   Safe replacement for bare grep -c output used in arithmetic.
# ---------------------------------------------------------------------------
_sum_lines() { awk '{for(i=1;i<=NF;i++) s+=$i} END{print s+0}'; }

# ---------------------------------------------------------------------------
# _grep_count CMD...
#   Run CMD, count matching lines, return a single clean integer.
#   Always succeeds (no exit-1 from grep finding nothing).
# ---------------------------------------------------------------------------
_grep_count() { { "$@" 2>/dev/null || true; } | wc -l | tr -d '[:space:]'; }

# ---------------------------------------------------------------------------
# _safe_wc FILE
#   wc -l on a file, return clean integer (no leading spaces).
# ---------------------------------------------------------------------------
_safe_wc() { wc -l < "${1:-/dev/null}" 2>/dev/null | tr -d '[:space:]' || echo "0"; }

# ---------------------------------------------------------------------------
# _parse_int EXPR
#   Parse the first integer from a string (useful for grep -oP output).
# ---------------------------------------------------------------------------
_parse_int() { echo "${1:-0}" | grep -oP '[0-9]+' | head -1 | tr -d '[:space:]' || echo "0"; }

# ---------------------------------------------------------------------------
# _max_int_from_cmd CMD...
#   Run CMD, extract all integers, return the maximum. Safe for temp/power.
# ---------------------------------------------------------------------------
_max_int_from_cmd() {
  { "$@" 2>/dev/null || true; } | grep -oP '[0-9]+' | \
    awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m}'
}

# Already defined in run_tests.sh but re-export so modules can use without sourcing run_tests
cmd_exists() { command -v "$1" &>/dev/null; }
