# scripts/start.sh
#!/bin/bash
# Start script for trading system

set -e

echo "Starting Trading System..."

# Check dependencies
echo "Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Check Redis
if ! redis-cli ping &> /dev/null; then
    echo "Warning: Redis not responding. Starting Redis..."
    redis-server --daemonize yes
    sleep 2
fi

# Validate configuration
echo "Validating configuration..."
python3 main.py validate-config

# Start the system
echo "Starting trading system..."
python3 main.py run