# strategies/__init__.py
from typing import Dict, Any
from core.interfaces import IStrategy

# Import your specific strategy classes
# from .moving_average import MovingAverageStrategy
# from .momentum import MomentumStrategy

def create_strategy(strategy_type: str, config: Dict[str, Any]) -> IStrategy:
    """Factory function to create strategy instances"""
    strategy_map = {
        'moving_average': MovingAverageStrategy,
        'momentum': MomentumStrategy,
        # Add other strategies here
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    strategy_class = strategy_map[strategy_type]
    return strategy_class(config)

# Export the function
__all__ = ['create_strategy']