#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
MAX_TIME="${MAX_TIME:-20}"
MESSAGE1="${MESSAGE1:-Recommend a phone with good battery life.}"
MESSAGE2="${MESSAGE2:-I care more about battery and camera.}"

if command -v curl >/dev/null 2>&1; then
  CURL_BIN="curl"
elif command -v curl.exe >/dev/null 2>&1; then
  CURL_BIN="curl.exe"
else
  echo "curl is required but not found."
  exit 1
fi

echo "[1/5] Health check: ${BASE_URL}/health"
HEALTH_RESPONSE="$("$CURL_BIN" -sS --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${MAX_TIME}" "${BASE_URL}/health")"
echo "  response: ${HEALTH_RESPONSE}"

echo "[2/5] Create session: ${BASE_URL}/sessions"
SESSION_RESPONSE="$("$CURL_BIN" -sS --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${MAX_TIME}" \
  -X POST "${BASE_URL}/sessions" \
  -H "Content-Type: application/json" \
  --data-raw "{}")"
echo "  response: ${SESSION_RESPONSE}"
SESSION_ID="$(
  printf "%s" "${SESSION_RESPONSE}" \
    | tr -d '\r\n' \
    | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
)"

if [[ -z "${SESSION_ID}" ]]; then
  echo "Failed to parse session_id from /sessions response."
  exit 1
fi
echo "  session_id: ${SESSION_ID}"

echo "[3/5] Chat round 1: /chat"
CHAT1_BODY='{"message":"'"${MESSAGE1}"'","session_id":"'"${SESSION_ID}"'","stream":false}'
CHAT1_RESPONSE="$(
  printf "%s" "${CHAT1_BODY}" | "$CURL_BIN" -sS --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${MAX_TIME}" \
  -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  --data-binary @-
)"
echo "  response: ${CHAT1_RESPONSE}"

echo "[4/5] Chat round 2: /chat"
CHAT2_BODY='{"message":"'"${MESSAGE2}"'","session_id":"'"${SESSION_ID}"'","stream":false}'
CHAT2_RESPONSE="$(
  printf "%s" "${CHAT2_BODY}" | "$CURL_BIN" -sS --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${MAX_TIME}" \
  -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  --data-binary @-
)"
echo "  response: ${CHAT2_RESPONSE}"

echo "[5/5] Get and delete session"
SESSION_DETAIL_RESPONSE="$("$CURL_BIN" -sS --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${MAX_TIME}" "${BASE_URL}/sessions/${SESSION_ID}?limit=20")"
echo "  detail: ${SESSION_DETAIL_RESPONSE}"
DELETE_RESPONSE="$("$CURL_BIN" -sS --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${MAX_TIME}" -X DELETE "${BASE_URL}/sessions/${SESSION_ID}")"
echo "  delete: ${DELETE_RESPONSE}"

echo "Done."
