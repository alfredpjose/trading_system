# core/exceptions.py
class TradingSystemError(Exception):
    """Base exception for trading system"""
    pass

class ConnectionError(TradingSystemError):
    """Broker/data connection errors"""
    pass

class OrderError(TradingSystemError):
    """Order execution errors"""
    pass

class RiskError(TradingSystemError):
    """Risk management violations"""
    pass

class StrategyError(TradingSystemError):
    """Strategy execution errors"""
    pass

class ConfigurationError(TradingSystemError):
    """Configuration errors"""
    pass