#!/bin/bash
# ToM-NAS: Start Script for NAS Deployment
#
# This script starts the complete ToM-NAS simulation environment
# on a repurposed laptop running Linux.

set -e

echo "=================================================="
echo "  ToM-NAS: Theory of Mind Simulation Environment"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Install Docker with: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Use docker compose v2 if available
COMPOSE_CMD="docker-compose"
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

# Get local IP address
get_local_ip() {
    ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || hostname -I | awk '{print $1}'
}

LOCAL_IP=$(get_local_ip)

echo -e "${YELLOW}Checking system requirements...${NC}"

# Check available memory
MEM_TOTAL=$(free -m | awk '/^Mem:/{print $2}')
if [ "$MEM_TOTAL" -lt 2048 ]; then
    echo -e "${YELLOW}Warning: Less than 2GB RAM available (${MEM_TOTAL}MB)${NC}"
    echo "The simulation may run slowly."
fi

# Check available disk space
DISK_FREE=$(df -m . | awk 'NR==2{print $4}')
if [ "$DISK_FREE" -lt 1024 ]; then
    echo -e "${YELLOW}Warning: Less than 1GB disk space available${NC}"
fi

echo -e "${GREEN}System check passed${NC}"
echo ""

# Build and start
echo -e "${YELLOW}Building containers...${NC}"
$COMPOSE_CMD -f docker-compose.nas.yml build

echo ""
echo -e "${YELLOW}Starting simulation environment...${NC}"
$COMPOSE_CMD -f docker-compose.nas.yml up -d

# Wait for services to be ready
echo ""
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 5

# Check service health
check_service() {
    local name=$1
    local container=$2
    if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
        echo -e "  ${GREEN}✓${NC} $name"
        return 0
    else
        echo -e "  ${RED}✗${NC} $name"
        return 1
    fi
}

echo "Service status:"
check_service "Simulation Engine" "tom-simulation"
check_service "WebSocket Bridge" "tom-bridge"
check_service "Web Interface" "tom-web"
check_service "Proxy" "tom-proxy"

echo ""
echo "=================================================="
echo -e "${GREEN}ToM-NAS is now running!${NC}"
echo "=================================================="
echo ""
echo "Access the simulation from any device on your network:"
echo ""
echo -e "  ${GREEN}http://${LOCAL_IP}${NC}"
echo ""
echo "Controls:"
echo "  WASD     - Move"
echo "  Mouse    - Look around"
echo "  E        - Interact"
echo "  Tab      - World map"
echo "  P        - Pause"
echo "  Escape   - Menu"
echo ""
echo "Useful commands:"
echo "  ./start-nas.sh          - Start the simulation"
echo "  ./stop-nas.sh           - Stop the simulation"
echo "  docker-compose -f docker-compose.nas.yml logs -f    - View logs"
echo ""
