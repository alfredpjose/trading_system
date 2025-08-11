# strategies/conventional/momentum.py
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from core.interfaces import MarketData, Signal, OrderAction
from strategies.base import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, strategy_id: str, event_bus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        
        # Momentum-specific parameters
        self.momentum_threshold = config.get('momentum_threshold', 0.02)
        self.rsi_period = config.get('rsi_period', 14)
        self.volume_threshold = config.get('volume_threshold', 1.5)  # Volume multiplier
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate momentum-based signal"""
        prices = self.get_price_array(data.symbol)
        
        if len(prices) < self.lookback_period:
            return None
        
        # Calculate momentum
        momentum = self._calculate_momentum(prices)
        
        # Calculate RSI for confirmation
        rsi = self._calculate_rsi(prices)
        
        # Check volume confirmation
        volume_confirmed = self._check_volume_confirmation(data)
        
        # Generate signal
        signal = None
        confidence = 0.0
        
        if momentum > self.momentum_threshold and rsi < 70 and volume_confirmed:
            # Strong upward momentum, not overbought, good volume
            confidence = min(momentum / self.momentum_threshold, 2.0) * 0.5
            if rsi < 50:  # Extra confidence if not even neutral
                confidence *= 1.2
            
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.BUY,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={
                    'momentum': momentum,
                    'rsi': rsi,
                    'price': data.price,
                    'volume_confirmed': volume_confirmed
                }
            )
        
        elif momentum < -self.momentum_threshold and rsi > 30 and volume_confirmed:
            # Strong downward momentum, not oversold, good volume
            confidence = min(abs(momentum) / self.momentum_threshold, 2.0) * 0.5
            if rsi > 50:  # Extra confidence if not even neutral
                confidence *= 1.2
            
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.SELL,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={
                    'momentum': momentum,
                    'rsi': rsi,
                    'price': data.price,
                    'volume_confirmed': volume_confirmed
                }
            )
        
        return signal
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum"""
        if len(prices) < self.lookback_period:
            return 0.0
        
        current_price = prices[-1]
        lookback_price = prices[-self.lookback_period]
        
        return (current_price - lookback_price) / lookback_price
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral RSI
        
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
    
    def _check_volume_confirmation(self, data: MarketData) -> bool:
        """Check if volume confirms the price movement"""
        if not data.volume or data.symbol not in self.price_history:
            return True  # Default to true if no volume data
        
        history = list(self.price_history[data.symbol])
        if len(history) < 5:
            return True
        
        # Get average volume of last 5 periods
        recent_volumes = [h.get('volume', 0) for h in history[-5:] if h.get('volume')]
        if not recent_volumes:
            return True
        
        avg_volume = np.mean(recent_volumes)
        
        # Current volume should be above threshold multiplier of average
        return data.volume >= (avg_volume * self.volume_threshold)