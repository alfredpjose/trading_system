"""
Trading System Configuration Settings
Handles loading and validation of configuration from YAML files
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.20
    position_sizing_method: str = "fixed_fractional"
    
    def __post_init__(self):
        if not 0 < self.max_daily_loss < 1:
            raise ValueError("max_daily_loss must be between 0 and 1")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be between 0 and 1")


@dataclass
class ExecutionConfig:
    """Execution configuration"""
    slippage_model: str = "linear"
    commission_rate: float = 0.001
    market_impact_factor: float = 0.0001
    order_timeout: int = 30
    
    def __post_init__(self):
        if self.commission_rate < 0:
            raise ValueError("commission_rate cannot be negative")
        if self.order_timeout <= 0:
            raise ValueError("order_timeout must be positive")


@dataclass
class DataRequirementsConfig:
    """Data requirements configuration"""
    min_history_days: int = 252
    required_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h", "1d"])
    required_indicators: List[str] = field(default_factory=lambda: ["sma", "ema", "rsi", "bollinger", "volume"])
    
    def __post_init__(self):
        if self.min_history_days <= 0:
            raise ValueError("min_history_days must be positive")


@dataclass
class GlobalSettingsConfig:
    """Global settings configuration"""
    max_concurrent_positions: int = 5
    portfolio_heat: float = 0.10
    correlation_threshold: float = 0.7
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data_requirements: DataRequirementsConfig = field(default_factory=DataRequirementsConfig)
    
    def __post_init__(self):
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if not 0 < self.portfolio_heat < 1:
            raise ValueError("portfolio_heat must be between 0 and 1")


@dataclass
class StrategyConfig:
    """Individual strategy configuration"""
    enabled: bool = False
    type: str = "conventional"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.type not in ["conventional", "ml"]:
            raise ValueError("Strategy type must be 'conventional' or 'ml'")
        
        # Validate common parameters
        if "risk_per_trade" in self.parameters:
            risk = self.parameters["risk_per_trade"]
            if not isinstance(risk, (int, float)) or not 0 < risk < 1:
                raise ValueError("risk_per_trade must be between 0 and 1")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create StrategyConfig from dictionary"""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")
        
        return cls(
            enabled=data.get("enabled", False),
            type=data.get("type", "conventional"),
            description=data.get("description", ""),
            parameters=data.get("parameters", {})
        )


@dataclass
class TradingHoursConfig:
    """Trading hours configuration"""
    start: str = "09:30"
    end: str = "16:00"
    timezone: str = "US/Eastern"


@dataclass
class AssetClassConfig:
    """Asset class configuration"""
    trading_hours: TradingHoursConfig = field(default_factory=TradingHoursConfig)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    major_pairs: Optional[List[str]] = None
    min_market_cap: Optional[float] = None
    exchanges: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssetClassConfig':
        """Create AssetClassConfig from dictionary"""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")
        
        trading_hours_data = data.get("trading_hours", {})
        trading_hours = TradingHoursConfig(
            start=trading_hours_data.get("start", "09:30"),
            end=trading_hours_data.get("end", "16:00"),
            timezone=trading_hours_data.get("timezone", "US/Eastern")
        )
        
        return cls(
            trading_hours=trading_hours,
            min_price=data.get("min_price"),
            max_price=data.get("max_price"),
            min_volume=data.get("min_volume"),
            major_pairs=data.get("major_pairs"),
            min_market_cap=data.get("min_market_cap"),
            exchanges=data.get("exchanges")
        )


@dataclass
class BacktestingConfig:
    """Backtesting configuration"""
    default_start_date: str = "2020-01-01"
    default_end_date: str = "2024-12-31"
    initial_capital: float = 100000
    benchmark: str = "SPY"
    metrics: List[str] = field(default_factory=lambda: [
        "total_return", "sharpe_ratio", "max_drawdown", 
        "win_rate", "profit_factor", "calmar_ratio"
    ])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestingConfig':
        """Create BacktestingConfig from dictionary"""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")
        
        return cls(
            default_start_date=data.get("default_start_date", "2020-01-01"),
            default_end_date=data.get("default_end_date", "2024-12-31"),
            initial_capital=data.get("initial_capital", 100000),
            benchmark=data.get("benchmark", "SPY"),
            metrics=data.get("metrics", [
                "total_return", "sharpe_ratio", "max_drawdown",
                "win_rate", "profit_factor", "calmar_ratio"
            ])
        )


@dataclass
class TradingSystemConfig:
    """Main trading system configuration"""
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    global_settings: GlobalSettingsConfig = field(default_factory=GlobalSettingsConfig)
    asset_classes: Dict[str, AssetClassConfig] = field(default_factory=dict)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    
    @classmethod
    def load_from_yaml(cls, yaml_path: Union[str, Path]) -> 'TradingSystemConfig':
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {yaml_path}: {e}")
        
        if not isinstance(data, dict):
            raise ValueError("YAML file must contain a dictionary at root level")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSystemConfig':
        """Create TradingSystemConfig from dictionary"""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")
        
        # Parse strategies
        strategies = {}
        strategies_data = data.get("strategies", {})
        if isinstance(strategies_data, dict):
            for name, strategy_data in strategies_data.items():
                if isinstance(strategy_data, dict):
                    strategies[name] = StrategyConfig.from_dict(strategy_data)
                else:
                    raise ValueError(f"Strategy '{name}' configuration must be a dictionary")
        
        # Parse global settings
        global_settings_data = data.get("global_settings", {})
        global_settings = cls._parse_global_settings(global_settings_data)
        
        # Parse asset classes
        asset_classes = {}
        asset_classes_data = data.get("asset_classes", {})
        if isinstance(asset_classes_data, dict):
            for name, asset_data in asset_classes_data.items():
                if isinstance(asset_data, dict):
                    asset_classes[name] = AssetClassConfig.from_dict(asset_data)
        
        # Parse backtesting
        backtesting_data = data.get("backtesting", {})
        backtesting = BacktestingConfig.from_dict(backtesting_data) if backtesting_data else BacktestingConfig()
        
        return cls(
            strategies=strategies,
            global_settings=global_settings,
            asset_classes=asset_classes,
            backtesting=backtesting
        )
    
    @staticmethod
    def _parse_global_settings(data: Dict[str, Any]) -> GlobalSettingsConfig:
        """Parse global settings from dictionary"""
        if not isinstance(data, dict):
            return GlobalSettingsConfig()
        
        # Parse risk management
        risk_mgmt_data = data.get("risk_management", {})
        risk_management = RiskManagementConfig()
        if isinstance(risk_mgmt_data, dict):
            risk_management = RiskManagementConfig(
                max_daily_loss=risk_mgmt_data.get("max_daily_loss", 0.05),
                max_drawdown=risk_mgmt_data.get("max_drawdown", 0.20),
                position_sizing_method=risk_mgmt_data.get("position_sizing_method", "fixed_fractional")
            )
        
        # Parse execution
        execution_data = data.get("execution", {})
        execution = ExecutionConfig()
        if isinstance(execution_data, dict):
            execution = ExecutionConfig(
                slippage_model=execution_data.get("slippage_model", "linear"),
                commission_rate=execution_data.get("commission_rate", 0.001),
                market_impact_factor=execution_data.get("market_impact_factor", 0.0001),
                order_timeout=execution_data.get("order_timeout", 30)
            )
        
        # Parse data requirements
        data_req_data = data.get("data_requirements", {})
        data_requirements = DataRequirementsConfig()
        if isinstance(data_req_data, dict):
            data_requirements = DataRequirementsConfig(
                min_history_days=data_req_data.get("min_history_days", 252),
                required_timeframes=data_req_data.get("required_timeframes", ["1m", "5m", "1h", "1d"]),
                required_indicators=data_req_data.get("required_indicators", ["sma", "ema", "rsi", "bollinger", "volume"])
            )
        
        return GlobalSettingsConfig(
            max_concurrent_positions=data.get("max_concurrent_positions", 5),
            portfolio_heat=data.get("portfolio_heat", 0.10),
            correlation_threshold=data.get("correlation_threshold", 0.7),
            risk_management=risk_management,
            execution=execution,
            data_requirements=data_requirements
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate strategies
        if not self.strategies:
            errors.append("No strategies configured")
        
        enabled_strategies = [name for name, config in self.strategies.items() if config.enabled]
        if not enabled_strategies:
            errors.append("No strategies are enabled")
        
        # Validate global settings
        if self.global_settings.max_concurrent_positions <= 0:
            errors.append("max_concurrent_positions must be positive")
        
        if not 0 < self.global_settings.portfolio_heat < 1:
            errors.append("portfolio_heat must be between 0 and 1")
        
        return errors
    
    def get_enabled_strategies(self) -> Dict[str, StrategyConfig]:
        """Get only enabled strategies"""
        return {name: config for name, config in self.strategies.items() if config.enabled}


# Default configuration paths
CONFIG_DIR = Path(__file__).parent
STRATEGIES_CONFIG_PATH = CONFIG_DIR / "strategies.yaml"

# Global config instance
_config: Optional[TradingSystemConfig] = None


def get_config() -> TradingSystemConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config(config_path: Optional[Union[str, Path]] = None) -> TradingSystemConfig:
    """Load configuration from file"""
    if config_path is None:
        config_path = STRATEGIES_CONFIG_PATH
    
    return TradingSystemConfig.load_from_yaml(config_path)


def reload_config() -> TradingSystemConfig:
    """Reload configuration from file"""
    global _config
    _config = load_config()
    return _config