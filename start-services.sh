#!/usr/bin/env bash
# Start the three processes the chat stack needs, in dependency order.
#   ollama :11434  ->  server1.py :8088 (MCP)  ->  mcp-main.py :8085 (web UI)
# Stop them with: ./start-services.sh stop
cd "$(dirname "$0")"
PY=/venv/main/bin/python
LOGDIR="${LOGDIR:-logs}"; mkdir -p "$LOGDIR"

pids_for() { ps -eo pid,args | grep -F "$1" | grep -v grep | awk '{print $1}'; }

stop_all() {
  for pat in "mcp-main.py" "server1.py" "ollama serve"; do
    for p in $(pids_for "$pat"); do echo "  stopping $pat (pid $p)"; kill "$p" 2>/dev/null; done
  done
}

[ "$1" = "stop" ] && { stop_all; exit 0; }

# 1. ollama, pointed at the cached model store
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "starting ollama..."
  setsid env OLLAMA_MODELS=/workspace/.ollama/models nohup ollama serve \
    > "$LOGDIR/ollama.log" 2>&1 < /dev/null &
  until curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 1; done
fi
echo "  ollama   :11434 up"

# 2. MCP tool server
if ! curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1; then
  echo "starting server1.py..."
  setsid nohup "$PY" server1.py --host 0.0.0.0 --port 8088 \
    > "$LOGDIR/server1.log" 2>&1 < /dev/null &
  until curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1; do sleep 1; done
fi
echo "  mcp srv  :8088  up"

# 3. web app (connects to the MCP server on startup, so it must come last).
# HTTPS once certs/vmukti.{pem,key} exist (ENABLE_HTTPS defaults to true in
# mcp-main.py) - -k because the cert's CN is *.vmukti.com, which 127.0.0.1
# does not match; the check only cares that TLS answers, not the hostname.
if ! curl -skf https://127.0.0.1:8085/api/health >/dev/null 2>&1; then
  echo "starting mcp-main.py..."
  setsid nohup "$PY" mcp-main.py > "$LOGDIR/mcp-main.log" 2>&1 < /dev/null &
  until curl -skf https://127.0.0.1:8085/api/health >/dev/null 2>&1; do sleep 1; done
fi
echo "  web app  :8085  up (https)"
