#!/usr/bin/env python3
"""
Windows Setup für Paper Trading Transition - Korrigierte Version
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🚀 Windows Paper Trading Setup")
    print("=" * 60)

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    packages = [
        "pandas", "numpy", "yfinance", 
        "openpyxl", "xlsxwriter", 
        "typer", "loguru", "pyyaml"
    ]
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, check=True)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    dirs = [
        "logs", "logs/system", "config", "scripts", 
        "execution", "data", "backups"
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def update_env_file():
    """Create/update .env file"""
    print("\n⚙️ Setting up .env file...")
    
    env_content = """# Paper Trading Configuration
USE_REAL_POSITIONS=true
REMOVE_MOCK_DATA=true
ENABLE_PAPER_TRADING=true
TRADING_MODE=paper
IBKR_PAPER_ACCOUNT=DUH802936
USE_REAL_DATA=true
ENABLE_ALL_MARKETS=true
MARKET_DATA_TYPE=delayed

# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1000

# Data Configuration
DATA_REDIS_URL=redis://localhost:6379
DATA_DATABASE_URL=sqlite+aiosqlite:///trading.db

# Risk Configuration
RISK_MAX_PORTFOLIO_RISK=0.02
RISK_MAX_POSITION_SIZE=0.1
RISK_MAX_DRAWDOWN=0.2

# System Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
"""
    
    with open(".env", "w", encoding='utf-8') as f:
        f.write(env_content)
    
    print("  ✅ .env file created")
    return True

def create_transition_template():
    """Create transition script template"""
    print("\n📄 Creating transition script template...")
    
    template_content = '''#!/usr/bin/env python3
"""
Paper Trading Transition Script Template
TODO: Copy the complete PaperTradingTransition class code here
"""

import os
import sys
from pathlib import Path
from datetime import datetime

print("🚀 Paper Trading Transition")
print("=" * 50)

# TODO: Add the complete PaperTradingTransition class from the artifacts here

def main():
    """Main transition function"""
    print("⚠️ Please copy the complete transition code from the artifacts")
    print("📋 Files needed:")
    print("  1. Copy PaperTradingTransition class to this file")
    print("  2. Add CLI commands to main.py")
    print("  3. Run this script again")
    
    print("\\n✅ Directory structure ready!")
    print("📁 Created directories for multi-market support")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("scripts/transition_to_paper.py")
    with open(script_path, "w", encoding='utf-8') as f:
        f.write(template_content)
    
    print(f"  ✅ Created: {script_path}")
    return True

def create_verification_template():
    """Create verification script template"""
    print("\n🔍 Creating verification script template...")
    
    verification_content = '''#!/usr/bin/env python3
"""
Paper Trading Verification Script Template
TODO: Copy the complete verification code here
"""

import os
import asyncio
from pathlib import Path

def verify_environment():
    """Verify environment setup"""
    print("🔧 Verifying Environment...")
    
    env_vars = [
        'USE_REAL_POSITIONS', 'ENABLE_PAPER_TRADING', 
        'TRADING_MODE', 'IBKR_HOST', 'IBKR_PORT'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        print(f"  {status} {var}: {value}")

def main():
    """Main verification"""
    print("🔍 Paper Trading Verification")
    print("=" * 40)
    
    verify_environment()
    
    print("\\n📋 Manual checks needed:")
    print("  1. IBKR TWS/Gateway running on port 7497")
    print("  2. Paper trading mode enabled")
    print("  3. Network connectivity")
    
    # TODO: Add complete verification code from artifacts

if __name__ == "__main__":
    main()
'''
    
    verification_path = Path("scripts/verify_paper_trading.py")
    with open(verification_path, "w", encoding='utf-8') as f:
        f.write(verification_content)
    
    print(f"  ✅ Created: {verification_path}")
    return True

def print_next_steps():
    """Print next steps"""
    print("\n" + "=" * 60)
    print("✅ WINDOWS SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 NEXT STEPS:")
    print("\n1️⃣ Copy Transition Code:")
    print("   - Open: scripts\\transition_to_paper.py")
    print("   - Replace TODO with complete PaperTradingTransition class")
    print("   - Get code from 'transition_script' artifact")
    
    print("\n2️⃣ Add CLI Commands:")
    print("   - Open: main.py")
    print("   - Add commands from 'main_py_additions' artifact")
    
    print("\n3️⃣ Run Transition:")
    print("   python scripts\\transition_to_paper.py")
    
    print("\n4️⃣ Start IBKR:")
    print("   - Open IBKR TWS or IB Gateway")
    print("   - Enable Paper Trading mode")
    print("   - Use port 7497 for paper trading")
    
    print("\n5️⃣ Verify Setup:")
    print("   python scripts\\verify_paper_trading.py")
    
    print("\n6️⃣ Start Trading:")
    print("   python main.py run")
    
    print("\n🌍 Markets Available After Setup:")
    print("   • Stocks: US & International")
    print("   • Forex: Major & Minor pairs")
    print("   • Commodities: Gold, Oil, Gas, Agriculture")
    print("   • Futures: Indices, Bonds")
    print("   • Crypto: BTC, ETH (if available)")
    
    print(f"\n📁 Working Directory: {Path.cwd()}")

def main():
    """Main setup function"""
    try:
        print_header()
        
        # Run setup steps
        if not install_dependencies():
            print("❌ Dependency installation failed")
            return False
        
        if not create_directories():
            print("❌ Directory creation failed")
            return False
        
        if not update_env_file():
            print("❌ Environment file setup failed")
            return False
        
        if not create_transition_template():
            print("❌ Transition template creation failed")
            return False
        
        if not create_verification_template():
            print("❌ Verification template creation failed")
            return False
        
        print_next_steps()
        
        input("\nPress Enter to finish...")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        input("Press Enter to exit...")
        return False

if __name__ == "__main__":
    main()