#!/data/data/com.termux/files/usr/bin/bash
# Keeps the embedder and search server alive. Termux has no service manager,
# so this loop is the supervisor. Started once at boot.
#
# Liveness is checked over HTTP, not by process name: a name match is both
# fragile (any shell whose command line mentions the script matches it) and
# weaker (a hung process still "exists").
MODEL="$HOME/models/nomic-embed-text-v1.5.Q4_K_M.gguf"
DB="$HOME/obsidian-rag-data/portable.db"
EMBED_URL="http://127.0.0.1:8090"

# Loopback only. The sole client (Claude inside the Debian proot) shares this
# device's network stack, so it reaches 127.0.0.1 directly. Binding the
# Tailscale address instead made the socket depend on a VPN-assigned IP that
# goes stale whenever the VPN reconnects, and exposed the API to the tailnet
# for no reason.
BIND_HOST="127.0.0.1"
SEARCH_URL="http://$BIND_HOST:8100"

# shellcheck source=/dev/null
[ -f "$HOME/.obsidian-rag-token" ] && . "$HOME/.obsidian-rag-token"
export PHONE_RAG_TOKEN

alive() { curl -s -m 3 "$1/health" >/dev/null 2>&1; }

while true; do
  if ! alive "$EMBED_URL"; then
    echo "$(date -Is) embedder down, starting"
    pkill -x llama-server 2>/dev/null
    sleep 1
    nohup llama-server -m "$MODEL" --embedding --no-direct-io \
      --host 127.0.0.1 --port 8090 -c 4096 -ub 2048 >> "$HOME/llama.log" 2>&1 &
    sleep 25
  fi

  if ! alive "$SEARCH_URL"; then
    echo "$(date -Is) search server down, starting"
    pkill -f "$HOME/phone_search_server.py" 2>/dev/null
    sleep 1
    nohup python "$HOME/phone_search_server.py" --db "$DB" \
      --host "$BIND_HOST" --port 8100 --embed-url "$EMBED_URL" \
      >> "$HOME/search.log" 2>&1 &
    sleep 10
  fi

  sleep 60
done
