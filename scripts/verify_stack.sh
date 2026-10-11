#!/usr/bin/env bash
# `pass` always succeeds, so `cond && pass || fail` is a safe if/else here.
# shellcheck disable=SC2015
# Verify the local observability stack end to end.
#
#   scripts/verify_stack.sh                 # health + provisioning checks
#   SERVICES="python-inference-demo cpp-inference-demo" scripts/verify_stack.sh --telemetry
#
# --telemetry additionally checks that each service in $SERVICES has metrics in
# Prometheus and traces in Tempo (run the demos first: `make demo`).
# Requires curl and python3. Exit code 0 only if every check passes.
set -uo pipefail

COLLECTOR_HEALTH="${COLLECTOR_HEALTH:-http://localhost:13133}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
TEMPO_URL="${TEMPO_URL:-http://localhost:3200}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_AUTH="${GRAFANA_AUTH:-admin:admin}"
SERVICES="${SERVICES:-python-inference-demo cpp-inference-demo}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
CHECK_TELEMETRY=0
[[ "${1:-}" == "--telemetry" ]] && CHECK_TELEMETRY=1

failures=0
pass() { printf '  [ok] %s\n' "$1"; }
fail() { printf '  [!!] %s\n' "$1"; failures=$((failures + 1)); }

wait_for() {  # name url expected-substring
  local deadline=$((SECONDS + WAIT_SECONDS)) body
  while (( SECONDS < deadline )); do
    body="$(curl -fsS --max-time 3 "$2" 2>/dev/null || true)"
    if [[ "$body" == *"$3"* ]]; then pass "$1 ($2)"; return 0; fi
    sleep 2
  done
  fail "$1 not ready at $2"
  return 1
}

json() {  # python expression evaluated on parsed stdin JSON as `d`
  python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"
}

echo "Service health"
wait_for "collector health_check" "$COLLECTOR_HEALTH" "Server available"
wait_for "prometheus ready" "$PROMETHEUS_URL/-/ready" "Ready"
wait_for "tempo ready" "$TEMPO_URL/ready" "ready"
wait_for "grafana health" "$GRAFANA_URL/api/health" '"database"'

echo "Prometheus scrape targets"
targets="$(curl -fsS "$PROMETHEUS_URL/api/v1/targets" 2>/dev/null)"
for job in otel-collector-exporter otel-collector; do
  state="$(printf '%s' "$targets" | json "next((t['health'] for t in d['data']['activeTargets'] if t['labels']['job']=='$job'), 'missing')" 2>/dev/null)"
  [[ "$state" == "up" ]] && pass "target $job is up" || fail "target $job is $state"
done

echo "Grafana provisioning"
for uid in prometheus tempo; do
  status="$(curl -fsS -u "$GRAFANA_AUTH" "$GRAFANA_URL/api/datasources/uid/$uid/health" 2>/dev/null | json "d.get('status')" 2>/dev/null)"
  [[ "$status" == "OK" ]] && pass "datasource $uid healthy" || fail "datasource $uid status: ${status:-unreachable}"
done
count="$(curl -fsS -u "$GRAFANA_AUTH" "$GRAFANA_URL/api/search?tag=inference-profiler" 2>/dev/null | json "len(d)" 2>/dev/null)"
[[ "${count:-0}" -ge 5 ]] && pass "$count dashboards provisioned" || fail "expected 5 dashboards, found ${count:-0}"

if (( CHECK_TELEMETRY )); then
  echo "Telemetry from services: $SERVICES"
  now="$(date +%s)"
  for svc in $SERVICES; do
    n="$(curl -fsS -G "$PROMETHEUS_URL/api/v1/query" \
          --data-urlencode "query=sum(inference_requests_total{job=\"$svc\"})" 2>/dev/null \
        | json "d['data']['result'][0]['value'][1] if d['data']['result'] else 0" 2>/dev/null)"
    [[ "${n%.*}" -gt 0 ]] 2>/dev/null && pass "prometheus: $svc reported ${n%.*} requests" \
      || fail "prometheus: no inference_requests_total for job=$svc"
    t="$(curl -fsS -G "$TEMPO_URL/api/search" \
          --data-urlencode "q={resource.service.name=\"$svc\" && name=\"inference.request\"}" \
          --data-urlencode "start=$((now - 3600))" --data-urlencode "end=$now" \
          --data-urlencode "limit=20" 2>/dev/null | json "len(d.get('traces', []))" 2>/dev/null)"
    [[ "${t:-0}" -gt 0 ]] && pass "tempo: $svc has ${t} recent request traces" \
      || fail "tempo: no inference.request traces for $svc"
  done
fi

echo
if (( failures )); then echo "FAILED: $failures check(s)"; exit 1; fi
echo "All checks passed. Grafana: $GRAFANA_URL (dashboards folder 'Inference Profiler')"
