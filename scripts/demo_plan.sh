#!/usr/bin/env bash
# Demo for the thumbnail-plan API:
#   publish a plan -> restriction changes the playable set -> plan withdrawn
#   -> republish -> a restriction order strictly inside an already hidden
#   range leaves the playable set unchanged -> old plan stays queryable.
#
# Prereq: service listening on ${BASE:-http://localhost:8000}
#   docker compose up --build        # or: uvicorn app.main:app --port 8000
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"

req() { # method path [json-body] -> body + status line
  local method="$1" path="$2" body="${3:-}"
  echo
  echo "### $method $path ${body:+ $body}"
  if [ -n "$body" ]; then
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path" \
      -H 'Content-Type: application/json' --data "$body"
  else
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path"
  fi
}

wait_plan() { # poll GET until the background solve finishes; print final body
  local path="/videos/demo-plan/thumbnail-plan"
  for _ in $(seq 1 200); do
    local body
    body="$(curl -s "$BASE$path")"
    if printf '%s' "$body" | grep -q '"pending":false'; then
      printf '%s\n' "$body"
      return 0
    fi
    sleep 0.05
  done
  echo "timed out waiting for the plan" >&2
  return 1
}

echo "=== 1. import 10 shots (all playable) ==="
curl -s -w $'\nHTTP %{http_code}\n' -X POST "$BASE/videos/demo-plan/shots" \
  -H 'Content-Type: application/json' --data @"$HERE/demo_shots.json"

echo
echo "=== 2. request 3 thumbnails, shots 1 and 7 mandatory (async) ==="
req POST /videos/demo-plan/thumbnail-plan \
  '{"thumbnail_count":3,"required_shots":[1,7]}'

echo
echo "=== 3. poll until published: 3 half-open segments + chosen shot numbers ==="
wait_plan

echo
echo "=== 4. hide [2,6): playable set changes -> published plan withdrawn ==="
req POST /videos/demo-plan/restrictions '{"order_id":"lic-A","start":2,"end":6}'
req GET /videos/demo-plan/thumbnail-plan

echo
echo "=== 5. publish again over the remaining playable shots ==="
req POST /videos/demo-plan/thumbnail-plan '{"thumbnail_count":2}'
wait_plan

echo
echo "=== 6. add lic-B [3,5), entirely inside lic-A's hidden range: the playable"
echo "        set is unchanged, so the published plan is STILL queryable ==="
req POST /videos/demo-plan/restrictions '{"order_id":"lic-B","start":3,"end":5}'
req GET /videos/demo-plan/thumbnail-plan

echo
echo "=== 7. revoke lic-A: [2,3) and [5,6) become playable again -> withdrawn ==="
req DELETE /videos/demo-plan/restrictions/lic-A
req GET /videos/demo-plan/thumbnail-plan

echo
echo "=== 8. revoke lic-B: [3,5) becomes playable again -> playable set changed,"
echo "        still no plan to show until a new one is published ==="
req DELETE /videos/demo-plan/restrictions/lic-B
req GET /videos/demo-plan/thumbnail-plan
