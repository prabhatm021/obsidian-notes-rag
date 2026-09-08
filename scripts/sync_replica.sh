#!/usr/bin/env bash
# Push a fresh portable index to the replica whenever it is reachable.
#
# State-based, not event-based: every run checks reachability and whether the
# index actually changed. A replica that was offline for a week simply syncs on
# its next reachable run, so no queue or missed-change tracking is needed.
set -uo pipefail

REPLICA_HOST="${REPLICA_HOST:-u0_a215@100.87.116.13}"
REPLICA_PORT="${REPLICA_PORT:-8022}"
REPLICA_DB_DIR="${REPLICA_DB_DIR:-obsidian-rag-data}"
SOURCE_DB="${SOURCE_DB:-$HOME/.local/share/obsidian-notes-rag/obsidian_notes.db}"
REPO_DIR="${REPO_DIR:-$HOME/Downloads/obsidian-notes-rag}"
STATE_FILE="${STATE_FILE:-$HOME/.local/state/obsidian-rag-sync.stamp}"
EXPORT_TMP="${EXPORT_TMP:-/tmp/obsidian-rag-portable.db}"

SSH_OPTS=(-p "$REPLICA_PORT" -o BatchMode=yes -o ConnectTimeout=8)

log() { echo "$(date -Is) $*"; }

if [[ ! -f "$SOURCE_DB" ]]; then
  log "no source index at $SOURCE_DB"
  exit 0
fi

if ! ssh "${SSH_OPTS[@]}" "$REPLICA_HOST" true 2>/dev/null; then
  log "replica unreachable, skipping (will retry next run)"
  exit 0
fi

# Skip the export when the index has not changed since the last successful sync.
source_stamp="$(stat -c '%Y:%s' "$SOURCE_DB")"
mkdir -p "$(dirname "$STATE_FILE")"
if [[ -f "$STATE_FILE" && "$(cat "$STATE_FILE")" == "$source_stamp" ]]; then
  log "index unchanged since last sync, nothing to do"
  exit 0
fi

log "exporting portable index"
if ! (cd "$REPO_DIR" && uv run obsidian-rag export-portable "$EXPORT_TMP" >/dev/null); then
  log "export failed"
  exit 1
fi

log "sending to replica"
if ! rsync -e "ssh ${SSH_OPTS[*]}" "$EXPORT_TMP" \
     "$REPLICA_HOST:~/$REPLICA_DB_DIR/portable.db" >/dev/null; then
  log "rsync failed"
  exit 1
fi

# The replica loads the index into memory at startup, so it keeps serving the
# old data until restarted. Only stop it here — the replica's supervisor owns
# starting it, so there is one owner of the process and no start race.
log "stopping replica search server so its supervisor reloads the new index"
ssh "${SSH_OPTS[@]}" "$REPLICA_HOST" 'pkill -f "[p]hone_search_server.py"' >/dev/null 2>&1

echo "$source_stamp" > "$STATE_FILE"
rm -f "$EXPORT_TMP"
log "sync complete"
