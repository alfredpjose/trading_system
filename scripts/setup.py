# scripts/setup.py
#!/usr/bin/env python3
"""
Setup script for trading system
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "logs/system",
        "logs/trades", 
        "logs/backtest",
        "data/cache",
        "models",
        "reports",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_config_files():
    """Create default configuration files"""
    
    # Create .env file
    env_content = """# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1000

# Data Configuration  
DATA_REDIS_URL=redis://localhost:6379
DATA_DATABASE_URL=sqlite:///trading.db

# Risk Configuration
RISK_MAX_PORTFOLIO_RISK=0.02
RISK_MAX_POSITION_SIZE=0.1
RISK_MAX_DRAWDOWN=0.2

# System Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
ENABLE_PAPER_TRADING=true
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("Created .env file")
    
    # Create strategies.yaml if it doesn't exist
    strategies_yaml = Path("config/strategies.yaml")
    if not strategies_yaml.exists():
        strategies_content = """strategies:
  - name: momentum_basic
    enabled: true
    capital_allocation: 0.4
    parameters:
      lookback_period: 20
      momentum_threshold: 0.02
      position_size: 100
      symbols: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
  
  - name: mean_reversion
    enabled: false
    capital_allocation: 0.3
    parameters:
      lookback_period: 10
      std_threshold: 2.0
      position_size: 50
      symbols: ['SPY', 'QQQ', 'IWM']
"""
        
        with open(strategies_yaml, "w") as f:
            f.write(strategies_content)
        print("Created config/strategies.yaml")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_redis():
    """Setup Redis if not running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✓ Redis is running")
    except:
        print("⚠ Redis not found. Please install and start Redis:")
        print("  - Ubuntu/Debian: sudo apt install redis-server")
        print("  - macOS: brew install redis && brew services start redis")
        print("  - Windows: Download from https://redis.io/download")

def main():
    """Main setup function"""
    print("Setting up Trading System...")
    
    create_directory_structure()
    create_config_files()
    install_dependencies()
    setup_redis()
    
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("1. Start TWS or IB Gateway")
    print("2. Ensure Redis is running")
    print("3. Run: python main.py validate-config")
    print("4. Run: python main.py run")

if __name__ == "__main__":
    main()

