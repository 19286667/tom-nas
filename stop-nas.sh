#!/bin/bash
# ToM-NAS: Stop Script

set -e

echo "Stopping ToM-NAS simulation..."

# Use docker compose v2 if available
COMPOSE_CMD="docker-compose"
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

$COMPOSE_CMD -f docker-compose.nas.yml down

echo "ToM-NAS stopped."
