#!/bin/bash
# ToM-NAS Webapp Launcher
# One-click script to start the web interface

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     ToM-NAS: Theory of Mind Neural Architecture Search   ║"
echo "║                    Web Application                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is available (either standalone or as plugin)
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Error: docker-compose is not installed.${NC}"
    echo "Please install docker-compose from https://docs.docker.com/compose/install/"
    exit 1
fi

# Parse arguments
REBUILD=false
DETACHED=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rebuild|-r) REBUILD=true ;;
        --detached|-d) DETACHED=true ;;
        --stop|-s)
            echo -e "${YELLOW}Stopping ToM-NAS webapp...${NC}"
            $COMPOSE_CMD down
            echo -e "${GREEN}Stopped!${NC}"
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./start-webapp.sh [options]"
            echo ""
            echo "Options:"
            echo "  --rebuild, -r    Rebuild the Docker image"
            echo "  --detached, -d   Run in background (detached mode)"
            echo "  --stop, -s       Stop the running webapp"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Build/Rebuild if needed
if [ "$REBUILD" = true ]; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    $COMPOSE_CMD build --no-cache
fi

# Start the webapp
echo -e "${GREEN}Starting ToM-NAS Web Application...${NC}"

if [ "$DETACHED" = true ]; then
    $COMPOSE_CMD up -d tom-nas-webapp
    echo ""
    echo -e "${GREEN}✓ Webapp started in background!${NC}"
else
    echo ""
    echo -e "${BLUE}The webapp will be available at:${NC}"
    echo -e "${GREEN}  ➜  http://localhost:8501${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    $COMPOSE_CMD up tom-nas-webapp
fi
