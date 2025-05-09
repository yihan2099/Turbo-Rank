#!/bin/bash
set -e

cd /workspace/turbo-rank

# Start the server with provided arguments or defaults
exec python -m deploy.server "$@"
