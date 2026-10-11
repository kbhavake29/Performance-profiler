#!/usr/bin/env bash
# Run the observability stack WITHOUT Docker, using locally installed binaries
# and the same configuration files as docker-compose.yml.
#
#   BIN_DIR=/path/to/binaries scripts/native_stack.sh start
#   scripts/native_stack.sh status
#   scripts/native_stack.sh stop
#
# BIN_DIR must contain: otelcol-contrib, prometheus, tempo, and a Grafana
# release directory (grafana-*/bin/grafana). Any of them may instead be on PATH.
# Ports match the Compose stack (4317/4318, 8889, 9090, 3200, 3000), except that
# Tempo's own OTLP receiver moves to 4417/4418 to avoid clashing with the collector.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$ROOT/.native-stack/bin}"
RUN_DIR="${RUN_DIR:-$ROOT/.native-stack/run}"
PID_FILE="$RUN_DIR/pids"

find_bin() {
  local name="$1"
  if [[ -x "$BIN_DIR/$name" ]]; then echo "$BIN_DIR/$name"; return; fi
  local found
  found="$(find -L "$BIN_DIR" -maxdepth 2 -type f -name "$name" -perm -u+x 2>/dev/null | head -n1 || true)"
  if [[ -n "$found" ]]; then echo "$found"; return; fi
  command -v "$name" || { echo "error: $name not found in $BIN_DIR or PATH" >&2; exit 1; }
}

grafana_home() {
  local dir
  for dir in "$BIN_DIR"/grafana*; do
    if [[ -x "$dir/bin/grafana" ]]; then echo "$dir"; return; fi
  done
  echo "error: Grafana release directory (grafana-*/bin/grafana) not found in $BIN_DIR" >&2
  exit 1
}

start() {
  if [[ -f "$PID_FILE" ]]; then echo "already running (see $PID_FILE); run '$0 stop' first"; exit 1; fi
  mkdir -p "$RUN_DIR"/{prometheus,tempo,grafana,logs}
  sed -e 's/otel-collector:/localhost:/g' -e 's/tempo:3200/localhost:3200/g' \
    "$ROOT/monitoring/prometheus.yml" > "$RUN_DIR/prometheus.yml"

  local otelcol prometheus tempo ghome
  otelcol="$(find_bin otelcol-contrib)"
  prometheus="$(find_bin prometheus)"
  tempo="$(find_bin tempo)"
  ghome="$(grafana_home)"
  : > "$PID_FILE"

  TEMPO_OTLP_GRPC_ENDPOINT=127.0.0.1:4417 TEMPO_OTLP_HTTP_ENDPOINT=127.0.0.1:4418 \
  TEMPO_DATA_DIR="$RUN_DIR/tempo" \
    nohup "$tempo" -config.file="$ROOT/monitoring/tempo.yaml" -config.expand-env=true \
    > "$RUN_DIR/logs/tempo.log" 2>&1 &
  echo "tempo $!" >> "$PID_FILE"

  TEMPO_OTLP_ENDPOINT=127.0.0.1:4417 \
  COLLECTOR_OTLP_GRPC_ENDPOINT=127.0.0.1:4317 COLLECTOR_OTLP_HTTP_ENDPOINT=127.0.0.1:4318 \
  COLLECTOR_PROMETHEUS_ENDPOINT=127.0.0.1:8889 COLLECTOR_HEALTH_ENDPOINT=127.0.0.1:13133 \
  COLLECTOR_TELEMETRY_HOST=127.0.0.1 \
    nohup "$otelcol" --config "$ROOT/monitoring/otel-collector-config.yaml" \
    > "$RUN_DIR/logs/otel-collector.log" 2>&1 &
  echo "otel-collector $!" >> "$PID_FILE"

  nohup "$prometheus" --config.file="$RUN_DIR/prometheus.yml" \
    --storage.tsdb.path="$RUN_DIR/prometheus" --web.listen-address=127.0.0.1:9090 \
    --enable-feature=exemplar-storage \
    > "$RUN_DIR/logs/prometheus.log" 2>&1 &
  echo "prometheus $!" >> "$PID_FILE"

  PROMETHEUS_URL=http://localhost:9090 TEMPO_URL=http://localhost:3200 \
  DASHBOARDS_DIR="$ROOT/monitoring/grafana/dashboards" \
  GF_PATHS_PROVISIONING="$ROOT/monitoring/grafana/provisioning" \
  GF_PATHS_DATA="$RUN_DIR/grafana" GF_PATHS_LOGS="$RUN_DIR/logs" \
  GF_SERVER_HTTP_ADDR=127.0.0.1 GF_SERVER_HTTP_PORT=3000 \
  GF_SECURITY_ADMIN_USER=admin GF_SECURITY_ADMIN_PASSWORD=admin \
  GF_AUTH_ANONYMOUS_ENABLED=true GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
  GF_ANALYTICS_REPORTING_ENABLED=false GF_ANALYTICS_CHECK_FOR_UPDATES=false \
  GF_NEWS_NEWS_FEED_ENABLED=false \
    nohup "$ghome/bin/grafana" server --homepath "$ghome" \
    > "$RUN_DIR/logs/grafana.log" 2>&1 &
  echo "grafana $!" >> "$PID_FILE"

  echo "started; logs in $RUN_DIR/logs. Check readiness with: scripts/verify_stack.sh"
}

stop() {
  if [[ ! -f "$PID_FILE" ]]; then echo "not running"; return; fi
  while read -r name pid; do
    if kill "$pid" 2>/dev/null; then echo "stopped $name ($pid)"; fi
  done < "$PID_FILE"
  rm -f "$PID_FILE"
}

status() {
  if [[ ! -f "$PID_FILE" ]]; then echo "not running"; return; fi
  while read -r name pid; do
    if kill -0 "$pid" 2>/dev/null; then echo "$name: running ($pid)"; else echo "$name: exited"; fi
  done < "$PID_FILE"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
