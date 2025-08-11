# scripts/stop.sh
#!/bin/bash
# Stop script for trading system

echo "Stopping Trading System..."

# Find and kill trading system processes
pkill -f "python.*main.py"

echo "Trading system stopped"