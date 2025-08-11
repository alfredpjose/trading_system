# config/__init__.py
from .settings import (
    TradingSystemConfig as SystemConfig,  # Alias for backward compatibility
    TradingSystemConfig, 
    StrategyConfig, 
    get_config,
    load_config,
    reload_config
)

# Add the missing functions here since they don't exist in settings.py yet
def load_strategy_configs():
    """Load strategy configurations"""
    config = get_config()
    enabled_strategies = config.get_enabled_strategies()
    
    # Convert to list of StrategyConfig objects with names
    strategy_list = []
    for name, strategy_config in enabled_strategies.items():
        # Add the name to the strategy config
        strategy_config.name = name
        # Add capital_allocation if it doesn't exist (default to equal weight)
        if not hasattr(strategy_config, 'capital_allocation'):
            strategy_config.capital_allocation = 1.0 / len(enabled_strategies)
        strategy_list.append(strategy_config)
    
    return strategy_list

def validate_system_config(config=None):
    """Validate system configuration"""
    if config is None:
        config = get_config()
    
    errors = config.validate()
    
    if errors:
        from loguru import logger
        for error in errors:
            logger.warning(f"Configuration warning: {error}")
        return False
    
    return True

__all__ = [
    'SystemConfig',  # Backward compatibility alias
    'TradingSystemConfig', 
    'StrategyConfig', 
    'get_config',
    'load_config',
    'reload_config',
    'load_strategy_configs', 
    'validate_system_config'
]