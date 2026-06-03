#!/bin/sh
set -e

# Fix ownership of mounted log directories.
# When Docker auto-creates bind-mount targets on the host they are root-owned;
# the app user (UID 1000) cannot write to them without this fix.
if [ -d "/app/logs" ]; then
    chown -R app:app /app/logs 2>/dev/null || true
fi

exec gosu app "$@"
