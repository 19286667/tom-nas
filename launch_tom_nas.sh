#!/bin/bash
# ToM-NAS GUI Launcher
# Double-click this file to start the application

cd "$(dirname "$0")"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found."
    echo "Please install Python 3 and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check for required packages
python3 -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip3 install torch --quiet
}

# Launch the GUI
echo "Starting ToM-NAS..."
python3 tom_nas_gui.py

