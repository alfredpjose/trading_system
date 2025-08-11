# config/settings.py
from pydantic import BaseSettings, Field, validator
from typing import Dict, List, Optional
import yaml
from pathlib import Path

class IBKRConfig(BaseSettings):
    """IBKR connection configuration"""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1000
    timeout: int = 30
    
    class Config:
        env_prefix = "IBKR_"

class RiskConfig(BaseSettings):
    """Risk management configuration"""
    max_portfolio_risk: float = Field(0.02, ge=0.01, le=0.1)  # 2% max risk per trade
    max_position_size: float = Field(0.1, ge=0.01, le=0.5)   # 10% max position
    max_drawdown: float = Field(0.2, ge=0.05, le=0.5)        # 20% max drawdown
    daily_loss_limit: float = Field(0.05, ge=0.01, le=0.2)   # 5% daily loss limit
    max_open_positions: int = Field(10, ge=1, le=50)
    
    class Config:
        env_prefix = "RISK_"

class DataConfig(BaseSettings):
    """Data management configuration"""
    cache_ttl: int = Field(60, ge=1, le=3600)  # Cache TTL in seconds
    redis_url: str = "redis://localhost:6379"
    database_url: str = "sqlite:///trading.db"
    max_historical_days: int = Field(252, ge=1, le=1000)
    
    class Config:
        env_prefix = "DATA_"

class SystemConfig(BaseSettings):
    """Main system configuration"""
    environment: str = Field("development", regex="^(development|staging|production)$")
    log_level: str = Field("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    enable_paper_trading: bool = True
    max_concurrent_strategies: int = Field(5, ge=1, le=20)
    heartbeat_interval: int = Field(30, ge=5, le=300)
    
    # Sub-configurations
    ibkr: IBKRConfig = IBKRConfig()
    risk: RiskConfig = RiskConfig()
    data: DataConfig = DataConfig()
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class StrategyConfig(BaseSettings):
    """Individual strategy configuration"""
    name: str
    enabled: bool = True
    capital_allocation: float = Field(0.1, ge=0.01, le=1.0)
    parameters: Dict = {}
    
    @validator('capital_allocation')
    def validate_allocation(cls, v):
        if not 0.01 <= v <= 1.0:
            raise ValueError('Capital allocation must be between 1% and 100%')
        return v

def load_strategy_configs(config_path: str = "config/strategies.yaml") -> List[StrategyConfig]:
    """Load strategy configurations from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Create default config
        default_config = {
            'strategies': [
                {
                    'name': 'momentum_basic',
                    'enabled': True,
                    'capital_allocation': 0.3,
                    'parameters': {
                        'lookback_period': 20,
                        'momentum_threshold': 0.02,
                        'position_size': 100
                    }
                },
                {
                    'name': 'mean_reversion',
                    'enabled': False,
                    'capital_allocation': 0.2,
                    'parameters': {
                        'lookback_period': 10,
                        'std_threshold': 2.0,
                        'position_size': 50
                    }
                }
            ]
        }
        
        config_file.parent.mkdir(exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return [StrategyConfig(**strategy) for strategy in config_data.get('strategies', [])]

def validate_system_config(config: SystemConfig) -> bool:
    """Validate system configuration for common issues"""
    errors = []
    
    # Check IBKR port ranges
    if config.ibkr.port not in [7496, 7497, 4001, 4002]:
        errors.append(f"Unusual IBKR port {config.ibkr.port}. Common ports: 7496, 7497, 4001, 4002")
    
    # Check risk parameters
    if config.risk.max_portfolio_risk > 0.05:
        errors.append("High portfolio risk setting (>5%) detected")
    
    # Check production settings
    if config.environment == "production":
        if config.enable_paper_trading:
            errors.append("Paper trading enabled in production environment")
        if config.log_level == "DEBUG":
            errors.append("Debug logging in production environment")
    
    if errors:
        for error in errors:
            print(f"WARNING: {error}")
        return False
    
    return True