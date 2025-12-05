#!/bin/bash
# ToM-NAS: One-command startup script
# Evolving AI that understands minds

echo "=============================================="
echo "  ToM-NAS: Theory of Mind Neural Architecture Search"
echo "  Evolving AI that understands minds"
echo "=============================================="
echo ""

# Check for Docker
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "Starting with Docker..."
    echo ""
    docker-compose up --build
else
    echo "Docker not found. Starting with Python..."
    echo ""

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is required but not installed."
        exit 1
    fi

    # Install dependencies if needed
    if [ ! -f ".deps_installed" ]; then
        echo "Installing dependencies (first time only)..."
        pip install -r requirements.txt
        touch .deps_installed
    fi

    # Start the server
    echo "Starting ToM-NAS web application..."
    echo "Open http://localhost:8000 in your browser"
    echo ""
    python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000
fi
