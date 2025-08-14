#!/usr/bin/env python3
"""
Clean up strategies configuration by removing problematic optimized strategies
and keeping only working base strategies
"""

import yaml
from pathlib import Path

def clean_strategies_config():
    """Remove problematic optimized strategies and keep only working ones"""
    
    config_path = Path("config/strategies.yaml")
    
    if not config_path.exists():
        print("❌ Config file not found")
        return False
    
    try:
        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("🔍 Current strategies:")
        for name, strategy in config.get('strategies', {}).items():
            enabled = strategy.get('enabled', False)
            strategy_type = strategy.get('type', 'unknown')
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            print(f"  {name} ({strategy_type}): {status}")
        
        # Remove problematic optimized strategies
        strategies_to_remove = []
        for name, strategy in config.get('strategies', {}).items():
            if name.startswith('optimized_') and strategy.get('type') == 'conventional':
                # These are causing issues because they're not mapped to actual strategy classes
                strategies_to_remove.append(name)
        
        if strategies_to_remove:
            print(f"\n🧹 Removing problematic strategies: {strategies_to_remove}")
            for name in strategies_to_remove:
                del config['strategies'][name]
        
        # Ensure we have working base strategies
        working_strategies = {
            "ma_crossover": {
                "enabled": True,
                "type": "conventional",
                "description": "Simple moving average crossover strategy",
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 30,
                    "asset_classes": ["stocks", "forex", "crypto"],
                    "risk_per_trade": 0.02,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.1,
                    "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
                }
            },
            "bollinger_bands": {
                "enabled": True,
                "type": "conventional", 
                "description": "Bollinger Bands breakout/reversion strategy",
                "parameters": {
                    "period": 20,
                    "std_dev": 2,
                    "mode": "breakout",
                    "asset_classes": ["stocks", "forex"],
                    "risk_per_trade": 0.02,
                    "stop_loss_pct": 0.04,
                    "take_profit_pct": 0.08,
                    "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
                }
            },
            "rsi_mean_reversion": {
                "enabled": False,
                "type": "conventional",
                "description": "RSI-based mean reversion strategy", 
                "parameters": {
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                    "asset_classes": ["stocks"],
                    "risk_per_trade": 0.015,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
                }
            }
        }
        
        # Update with working strategies
        config['strategies'] = working_strategies
        
        # Save cleaned config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print("\n✅ Configuration cleaned successfully!")
        print("\n📊 Active strategies:")
        for name, strategy in working_strategies.items():
            if strategy.get('enabled', False):
                symbols = strategy['parameters'].get('symbols', [])
                print(f"  ✅ {name}: {len(symbols)} symbols")
        
        # Validate the cleaned config
        with open(config_path, 'r') as f:
            yaml.safe_load(f)
        
        print("✅ Validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to clean config: {e}")
        return False

def main():
    """Main function"""
    print("🧹 Cleaning strategies configuration...")
    
    if clean_strategies_config():
        print("\n🎉 Configuration is now clean and ready!")
        print("\n📋 Next steps:")
        print("   1. Run: python main.py validate-config")
        print("   2. Run: python main.py run")
    else:
        print("❌ Failed to clean configuration")

if __name__ == "__main__":
    main()