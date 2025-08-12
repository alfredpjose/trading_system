# strategies/__init__.py - Complete implementation
from typing import Dict, Any
from core.interfaces import IStrategy, IEventBus
from loguru import logger
import numpy as np

# Import base strategy
from .base import BaseStrategy

def create_strategy(strategy_type: str, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]) -> IStrategy:
    """Factory function to create strategy instances"""
    
    # Map strategy names from config to actual classes
    strategy_map = {
        'ma': MovingAverageStrategy,
        'bollinger': BollingerBandsStrategy,
        'momentum': MomentumStrategy,
        'rsi': RSIMeanReversionStrategy,
        'lstm': LSTMStrategy,
        'rf': RandomForestStrategy,
    }
    
    if strategy_type not in strategy_map:
        # For strategies not yet implemented, create a basic strategy
        logger.warning(f"Strategy type '{strategy_type}' not found, using BaseStrategy")
        return BaseStrategy(strategy_id, event_bus, config)
    
    strategy_class = strategy_map[strategy_type]
    return strategy_class(strategy_id, event_bus, config)


# Import required modules for strategies
from datetime import datetime
from typing import Optional
from core.interfaces import MarketData, Signal, OrderAction


class MovingAverageStrategy(BaseStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 30)
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Default symbols
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate MA crossover signal"""
        prices = self.get_price_array(data.symbol)
        
        if len(prices) < self.slow_period:
            return None
        
        # Calculate moving averages
        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])
        
        # Previous MA values for crossover detection
        if len(prices) < self.slow_period + 1:
            return None
            
        prev_fast_ma = np.mean(prices[-(self.fast_period + 1):-1])
        prev_slow_ma = np.mean(prices[-(self.slow_period + 1):-1])
        
        signal = None
        
        # Bullish crossover: fast MA crosses above slow MA
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            confidence = min((fast_ma - slow_ma) / slow_ma * 10, 1.0)
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.BUY,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma, 'price': data.price}
            )
        
        # Bearish crossover: fast MA crosses below slow MA
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            confidence = min((slow_ma - fast_ma) / slow_ma * 10, 1.0)
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.SELL,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma, 'price': data.price}
            )
        
        return signal


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2)
        self.mode = config.get('mode', 'breakout')  # 'breakout' or 'reversion'
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate Bollinger Bands signal"""
        prices = self.get_price_array(data.symbol)
        
        if len(prices) < self.period:
            return None
        
        # Calculate Bollinger Bands
        sma = np.mean(prices[-self.period:])
        std = np.std(prices[-self.period:])
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        current_price = data.price
        signal = None
        
        if self.mode == 'breakout':
            # Breakout strategy: buy on upper band break, sell on lower band break
            if current_price > upper_band:
                confidence = min((current_price - upper_band) / upper_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.BUY,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
            elif current_price < lower_band:
                confidence = min((lower_band - current_price) / lower_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.SELL,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
        
        elif self.mode == 'reversion':
            # Mean reversion: sell at upper band, buy at lower band
            if current_price > upper_band:
                confidence = min((current_price - upper_band) / upper_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.SELL,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
            elif current_price < lower_band:
                confidence = min((lower_band - current_price) / lower_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.BUY,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
        
        return signal


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate RSI mean reversion signal"""
        prices = self.get_price_array(data.symbol)
        
        if len(prices) < self.rsi_period + 1:
            return None
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices)
        signal = None
        
        # Oversold condition - potential buy signal
        if rsi < self.oversold_threshold:
            confidence = (self.oversold_threshold - rsi) / self.oversold_threshold
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.BUY,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'rsi': rsi, 'price': data.price}
            )
        
        # Overbought condition - potential sell signal
        elif rsi > self.overbought_threshold:
            confidence = (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.SELL,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'rsi': rsi, 'price': data.price}
            )
        
        return signal
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


# Placeholder strategies for ML models
class LSTMStrategy(BaseStrategy):
    """LSTM Strategy Placeholder"""
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        logger.info(f"LSTM strategy processing {data.symbol} - not implemented yet")
        return None


class RandomForestStrategy(BaseStrategy):
    """Random Forest Strategy Placeholder"""
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        logger.info(f"Random Forest strategy processing {data.symbol} - not implemented yet")
        return None


# Import the momentum strategy from the existing file
from .conventional.momentum import MomentumStrategy

__all__ = ['create_strategy']