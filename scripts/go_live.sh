#!/usr/bin/env bash
# Make the project live from this machine (no GPU needed):
#   CPU model + guardrail gateway in Docker -> public Cloudflare tunnel ->
#   the Vercel deployment pointed at that tunnel, so the vercel.app URL answers.
#
#   scripts/go_live.sh           start (or repair) once
#   scripts/go_live.sh --watch   then keep checking; if the tunnel drops, make a
#                                new one and repoint Vercel (used by START.bat)
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker-compose.local.yml --profile public)
PROJECT=$(sed -n 's/.*"projectName":"\([^"]*\)".*/\1/p' .vercel/project.json 2>/dev/null || true)
SITE="https://${PROJECT:-project-2-qlora-guardrails-pipeline}.vercel.app"

say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
tunnel_url() {
  "${COMPOSE[@]}" logs tunnel 2>/dev/null \
    | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1 || true
}
answers() { curl -sf -m 20 "$1/ready" >/dev/null 2>&1; }

# Steps use explicit `|| return 1`: set -e is ignored inside functions called from `if`/`||`.
start_model() {
  "${COMPOSE[@]}" up -d --build llm gateway >/dev/null 2>&1 \
    || { say "Docker could not start the containers. Is Docker Desktop running?" >&2; return 1; }
  for _ in $(seq 1 120); do                       # first run downloads the model
    answers http://localhost:9000 && return 0
    sleep 5
  done
  say "The model did not start. Check: ${COMPOSE[*]} logs llm" >&2
  return 1
}

new_tunnel() {
  "${COMPOSE[@]}" up -d --force-recreate tunnel >/dev/null 2>&1 \
    || { say "Docker could not start the tunnel." >&2; return 1; }
  url=""
  for _ in $(seq 1 60); do
    url=$(tunnel_url); [ -n "$url" ] && break; sleep 2
  done
  [ -n "$url" ] || { say "No tunnel address yet. Check: ${COMPOSE[*]} logs tunnel" >&2; return 1; }
  for _ in $(seq 1 30); do answers "$url" && break; sleep 2; done
  say "Updating your Vercel site to use the new tunnel..."
  if ! { vercel env add VLLM_BASE_URL production --value "$url" --force --yes \
         && vercel env add RATE_LIMIT_PER_MIN production --value 10 --force --yes \
         && vercel deploy --prod --yes; } >/dev/null 2>&1; then
    say "Vercel update failed. Run 'vercel login', then try again." >&2
    return 1
  fi
}

make_live() {
  say "[1/3] Starting the AI model and gateway..."
  start_model || return 1
  say "[2/3] Opening the public link..."
  url=$(tunnel_url)
  if [ -n "$url" ] && answers "$url" && answers "$SITE"; then
    say "      Already open: $url"
  else
    new_tunnel || return 1
  fi
  say "[3/3] Checking your site..."
  for _ in $(seq 1 12); do answers "$SITE" && break; sleep 5; done
  if answers "$SITE"; then
    echo
    echo "  ======================================================"
    echo "   YOUR PROJECT IS LIVE"
    echo "   $SITE"
    echo "  ======================================================"
    echo
  else
    say "The site is not answering yet. Run this again in a minute." >&2
    return 1
  fi
}

make_live || [ "${1:-}" = "--watch" ]   # in watch mode, keep going and retry below

if [ "${1:-}" = "--watch" ]; then
  explorer.exe "$SITE" >/dev/null 2>&1 || true
  echo "  Keep this window open: it checks the link every minute"
  echo "  and repairs it by itself. To stop everything, run STOP.bat."
  echo
  misses=0
  while true; do
    sleep 60
    if answers "$SITE"; then misses=0; continue; fi
    misses=$((misses + 1))
    [ "$misses" -lt 2 ] && continue                # ignore one-off blips
    say "The public link stopped answering. Repairing..."
    if make_live; then misses=0; fi
  done
fi
