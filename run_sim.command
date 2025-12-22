#!/bin/bash
# Double-click this file to run the missile simulation

cd "$(dirname "$0")"

# Activate virtual environment and run
source venv/bin/activate
python3 Misslesim1draft.py

echo ""
echo "Press any key to close..."
read -n 1
